from .simple_tokenizer import SimpleTokenizer as _Tokenizer

import torch
import torch.nn as nn
import math

from typing import Union, List
from pkg_resources import packaging

from .bayes_pfl.flows import Planar
from .bayes_pfl.pfl import Encoder, Decoder, PlanarPFL, PlanarPFLState, binary_loss_function

# 0. [前置] 分词函数 Crane-main/models/simple_tokenizer.py
_tokenizer = _Tokenizer()
def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)
    ----------
    Parameters
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize
    context_length : int
        The context length to use; all CLIP models use 77 as the context length
    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length
    -------
    Returns
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """

    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]

    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result



# 动态生成可学习的文本提示 类CoOp
class PromptLearner(nn.Module):
    #1
    def __init__(self, clip_model, design_details):
        super().__init__()
        dtype = clip_model.transformer.get_cast_dtype()
        args = design_details["others"]
        # 获取配置参数
        self.train_with_img_cls_type = args.train_with_img_cls_type
        self.train_with_img_cls_prob = args.train_with_img_cls_prob
        self.text_encoder_n_ctx = design_details["learnabel_text_embedding_length"] 
        self.n_ctx = design_details["Prompt_length"]
        self.n_ctx_pos = self.n_ctx
        self.n_ctx_neg = self.n_ctx
        # 定义类别和模板
        self.classnames = ["object"]
        self.n_cls = len(self.classnames)

        self.use_bayes_prompt = bool(getattr(args, "use_bayes_prompt", False))
        if self.use_bayes_prompt:
            from .state_prompts import NORMAL_STATE_TEMPLATES, ABNORMAL_STATE_TEMPLATES
            self.state_normal_list = list(NORMAL_STATE_TEMPLATES)
            self.state_anomaly_list = list(ABNORMAL_STATE_TEMPLATES)
        else:
            self.state_normal_list = ["{}"]
            self.state_anomaly_list = ["damaged {}"]
        self.normal_num = len(self.state_normal_list)
        self.anormaly_num = len(self.state_anomaly_list)

        self.bayes_flow_steps = int(getattr(args, "bayes_flow_steps", 4))
        self.bayes_condition_on_image = bool(getattr(args, "bayes_condition_on_image", True))
        self.bayes_flow_type = getattr(args, "bayes_flow_type", "planar")
        self.bayes_align_official = bool(getattr(args, "bayes_align_official", True))
        self.bayes_kl_weight = float(getattr(args, "bayes_kl_weight", 0.01))
        self._last_kl = None
        self._last_pfl_loss = None
        self._cached_state_n = None
        self._cached_state_a = None

        ### 初始化 深层可学习提示
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.compound_prompts_depth = design_details["learnabel_text_embedding_depth"]
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(self.text_encoder_n_ctx, ctx_dim))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            # print("single_para", single_para.shape)
            nn.init.normal_(single_para, std=0.02)

        ### 初始化 浅层可学习提示
        ctx_vectors_pos = torch.empty(self.n_cls, self.normal_num, self.n_ctx_pos, ctx_dim, dtype=dtype)
        ctx_vectors_neg = torch.empty(self.n_cls, self.anormaly_num, self.n_ctx_pos, ctx_dim, dtype=dtype)
        ctx_init = str(getattr(args, "ctx_init", "random"))
        if ctx_init == "zeros":
            nn.init.zeros_(ctx_vectors_pos)
            nn.init.zeros_(ctx_vectors_neg)
        elif ctx_init == "clip":
            phrase = str(getattr(args, "ctx_init_phrase", "a photo of a"))
            token_device = clip_model.token_embedding.weight.device
            tokenized = tokenize(phrase).to(device=token_device)
            with torch.no_grad():
                emb = clip_model.token_embedding(tokenized).type(dtype)  # (1, 77, C)

            # Extract "content" token embeddings between SOT and EOT.
            eot_token = _tokenizer.encoder["<|endoftext|>"]
            eot_pos = int((tokenized[0] == eot_token).nonzero(as_tuple=False)[0].item())
            content = emb[0, 1:eot_pos, :]  # (T, C)
            if content.numel() == 0:
                nn.init.normal_(ctx_vectors_pos, std=0.02)
                nn.init.normal_(ctx_vectors_neg, std=0.02)
            else:
                # Tile/pad to n_ctx.
                rep = int(math.ceil(float(self.n_ctx_pos) / float(content.shape[0])))
                base = content.repeat(rep, 1)[: self.n_ctx_pos, :]  # (n_ctx, C)
                base_pos = base.view(1, 1, self.n_ctx_pos, ctx_dim)
                ctx_vectors_pos.copy_(base_pos.expand(self.n_cls, self.normal_num, -1, -1))
                base_neg = base.view(1, 1, self.n_ctx_neg, ctx_dim)
                ctx_vectors_neg.copy_(base_neg.expand(self.n_cls, self.anormaly_num, -1, -1))
        else:
            nn.init.normal_(ctx_vectors_pos, std=0.02)
            nn.init.normal_(ctx_vectors_neg, std=0.02)
        self.ctx_pos = nn.Parameter(ctx_vectors_pos)  # to be optimized
        self.ctx_neg = nn.Parameter(ctx_vectors_neg)  # to be optimized

        # Bayesian Prompt Flow (lightweight, normal-only friendly):
        # - model context tokens as Gaussian (mu=ctx_*, sigma=exp(logstd_*))
        # - optional image-conditioned mean shift (ISD) via a tiny projection
        # - simple flow-like refinement via residual steps (K)
        if self.use_bayes_prompt:
            init_logstd = float(getattr(args, "bayes_init_logstd", math.log(0.02)))
            self.ctx_pos_logstd = nn.Parameter(torch.full_like(self.ctx_pos, init_logstd))
            self.ctx_neg_logstd = nn.Parameter(torch.full_like(self.ctx_neg, init_logstd))

            # Residual gate: keep stochastic Bayes prompt as a small "delta" on top of
            # the deterministic (conditioned) mean to avoid destroying CLIP's manifold.
            self.bayes_use_residual = bool(getattr(args, "bayes_use_residual", True))
            alpha_init = float(getattr(args, "bayes_residual_alpha_init", 0.01))
            self.bayes_residual_alpha_pos = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
            self.bayes_residual_alpha_neg = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

            ctx_dim = clip_model.ln_final.weight.shape[0]
            self.ctx_cond_pos = nn.Linear(ctx_dim, ctx_dim, bias=True)
            self.ctx_cond_neg = nn.Linear(ctx_dim, ctx_dim, bias=True)
            nn.init.zeros_(self.ctx_cond_pos.weight)
            nn.init.zeros_(self.ctx_cond_neg.weight)
            nn.init.zeros_(self.ctx_cond_pos.bias)
            nn.init.zeros_(self.ctx_cond_neg.bias)

            if self.bayes_flow_type == "planar":
                self.planar_flows = nn.ModuleList([Planar() for _ in range(self.bayes_flow_steps)])
                # amortized flow params conditioned on image embedding (ISD-style)
                self.amor_u = nn.Linear(ctx_dim, self.bayes_flow_steps * ctx_dim)
                self.amor_w = nn.Linear(ctx_dim, self.bayes_flow_steps * ctx_dim)
                self.amor_b = nn.Linear(ctx_dim, self.bayes_flow_steps)
                nn.init.zeros_(self.amor_u.weight)
                nn.init.zeros_(self.amor_w.weight)
                nn.init.zeros_(self.amor_b.weight)
                nn.init.zeros_(self.amor_u.bias)
                nn.init.zeros_(self.amor_w.bias)
                nn.init.zeros_(self.amor_b.bias)
            else:
                self.prompt_flow = nn.ModuleList([nn.Linear(ctx_dim, ctx_dim) for _ in range(self.bayes_flow_steps)])
                for layer in self.prompt_flow:
                    nn.init.normal_(layer.weight, std=0.02)
                    nn.init.zeros_(layer.bias)

            # Bayes-PFL official-style alignment: PlanarPFL (image-specific) + PlanarPFLState (image-agnostic)
            if self.bayes_align_official:
                hidden = max(64, ctx_dim // 2)
                self._pfl_embed_dim = ctx_dim

                self._pfl_ctx_encoder = Encoder(ctx_dim, hidden, ctx_dim)
                self._pfl_ctx_decoder = Decoder(ctx_dim, hidden, ctx_dim)
                self._pfl_state_encoder = Encoder(ctx_dim, hidden, ctx_dim)
                self._pfl_state_decoder = Decoder(ctx_dim, hidden, ctx_dim)

                self.pfl_context = PlanarPFL(self._pfl_ctx_encoder, self._pfl_ctx_decoder, embed_dim=ctx_dim, num_flows=self.bayes_flow_steps)
                self.pfl_state_normal = PlanarPFLState(self._pfl_state_encoder, self._pfl_state_decoder, embed_dim=ctx_dim, num_flows=self.bayes_flow_steps)
                self.pfl_state_abnormal = PlanarPFLState(self._pfl_state_encoder, self._pfl_state_decoder, embed_dim=ctx_dim, num_flows=self.bayes_flow_steps)

                # Match Bayes-PFL initialization convention (normal vs abnormal separated)
                with torch.no_grad():
                    self.pfl_state_normal.state.normal_(mean=0.5, std=0.02)
                    self.pfl_state_abnormal.state.normal_(mean=-0.5, std=0.02)


        # NOTE: removing class description index
        ###  构造提示模板字符串
        prompt_prefix_pos = [" ".join(["X"] * self.n_ctx_pos)] * self.normal_num
        prompt_prefix_neg = [" ".join(["X"] * self.n_ctx_neg)] * self.anormaly_num  
        classnames = [name.replace("_", " ") for name in self.classnames]
        prompts_pos = [prompt_prefix_pos[idx] +  " " + template.format(name)+ "." for idx, template in enumerate(self.state_normal_list) for name in classnames]
        prompts_neg = [prompt_prefix_neg[idx] +  " " + template.format(name)+ "." for idx, template in enumerate(self.state_anomaly_list) for name in classnames]

        # Tokenize 并获取 Embedding
        tokenized_prompts_pos = [tokenize(p_pos) for p_pos in prompts_pos]
        tokenized_prompts_neg = [tokenize(p_neg) for p_neg in prompts_neg]
        tokenized_prompts_pos = torch.cat(tokenized_prompts_pos) # 'X X X X X X X X X X X X object.'
        tokenized_prompts_neg = torch.cat(tokenized_prompts_neg) # 'X X X X X X X X X X X X damaged object.'

        # Keep tokenized prompts on the same device as CLIP token embedding to avoid
        # CPU/CUDA mismatch in inference (test.py constructs PromptLearner with model on CUDA).
        token_device = clip_model.token_embedding.weight.device
        tokenized_prompts_pos = tokenized_prompts_pos.to(device=token_device)
        tokenized_prompts_neg = tokenized_prompts_neg.to(device=token_device)
        with torch.no_grad():
            embedding_pos = clip_model.token_embedding(tokenized_prompts_pos).type(dtype)
            embedding_neg = clip_model.token_embedding(tokenized_prompts_neg).type(dtype)
            n, l, d = embedding_pos.shape
            # print("embedding_pos", embedding_pos.shape)
            embedding_pos = embedding_pos.reshape(self.normal_num, self.n_cls, l, d).permute(1, 0, 2, 3)
            embedding_neg = embedding_neg.reshape(self.anormaly_num, self.n_cls, l, d).permute(1, 0, 2, 3)

        n, d = tokenized_prompts_pos.shape
        tokenized_prompts_pos = tokenized_prompts_pos.reshape(self.normal_num, self.n_cls, d).permute(1, 0, 2)
        n, d = tokenized_prompts_neg.shape
        tokenized_prompts_neg = tokenized_prompts_neg.reshape(self.anormaly_num, self.n_cls, d).permute(1, 0, 2)

        self.register_buffer("tokenized_prompts_pos", tokenized_prompts_pos)
        self.register_buffer("tokenized_prompts_neg", tokenized_prompts_neg)
        self.register_buffer("token_prefix_pos", embedding_pos[:, :, :1, :])
        self.register_buffer("token_suffix_pos", embedding_pos[:, :, 1 + self.n_ctx_pos:, :])
        self.register_buffer("token_prefix_neg", embedding_neg[:, :, :1, :])
        self.register_buffer("token_suffix_neg", embedding_neg[:, :, 1 + self.n_ctx_neg:, :])
        # print("tokenized_prompts shape", self.tokenized_prompts_pos.shape, self.tokenized_prompts_neg.shape)
        self.constructed_prompts = None

    def bayes_kl_loss(self):
        return self._last_kl

    def bayes_pfl_loss(self):
        return self._last_pfl_loss

    def _maybe_compute_pfl_biases(self, img_emb):
        if not (self.use_bayes_prompt and self.bayes_align_official and img_emb is not None):
            self._last_pfl_loss = None
            return None, None

        x = img_emb.float()
        # image-specific distribution
        x_recon, z_mu, z_std, log_det_j, z0, zk = self.pfl_context(x)
        _, rec, kl = binary_loss_function(
            x_recon,
            x,
            z_mu,
            z_std,
            z0,
            zk,
            log_det_j,
            z_size=self._pfl_embed_dim,
            beta=0.0,
            if_rec=True,
        )

        # image-agnostic distributions (cache in eval mode for stability, similar to official code)
        if self.training or self._cached_state_n is None or self._cached_state_a is None:
            x_recon_n, z_mu_n, z_std_n, log_det_j_n, z0_n, zk_n = self.pfl_state_normal()
            x_recon_a, z_mu_a, z_std_a, log_det_j_a, z0_a, zk_a = self.pfl_state_abnormal()
            if not self.training:
                self._cached_state_n = zk_n.detach()
                self._cached_state_a = zk_a.detach()
        else:
            # Recompute for loss terms? In eval we don't need loss; keep deterministic state samples.
            x_recon_n = z_mu_n = z_std_n = log_det_j_n = z0_n = None
            x_recon_a = z_mu_a = z_std_a = log_det_j_a = z0_a = None
            zk_n = self._cached_state_n.to(device=img_emb.device, dtype=img_emb.dtype)
            zk_a = self._cached_state_a.to(device=img_emb.device, dtype=img_emb.dtype)

        if self.training:
            # For state distributions we follow the official "if_rec=False": regularize only.
            _, _rec_n, kl_n = binary_loss_function(
                x_recon_n,
                self.pfl_state_normal.state,
                z_mu_n,
                z_std_n,
                z0_n,
                zk_n,
                log_det_j_n,
                z_size=self._pfl_embed_dim,
                beta=0.0,
                if_rec=False,
            )
            _, _rec_a, kl_a = binary_loss_function(
                x_recon_a,
                self.pfl_state_abnormal.state,
                z_mu_a,
                z_std_a,
                z0_a,
                zk_a,
                log_det_j_a,
                z_size=self._pfl_embed_dim,
                beta=0.0,
                if_rec=False,
            )

            self._last_pfl_loss = rec + self.bayes_kl_weight * (kl + kl_n + kl_a)
        else:
            self._last_pfl_loss = None

        # biases: context uses zk (image-specific), state uses zk_n/zk_a (image-agnostic)
        zk = zk.to(device=img_emb.device, dtype=img_emb.dtype)
        zk_n = zk_n.to(device=img_emb.device, dtype=img_emb.dtype)
        zk_a = zk_a.to(device=img_emb.device, dtype=img_emb.dtype)
        return zk, (zk_n, zk_a)

    def _bayes_sample_ctx(self, ctx_mean, ctx_logstd, img_emb, cond_proj, shift_vec=None, alpha_param=None):
        # Expand logstd to match ctx_mean shape
        logstd = ctx_logstd
        if ctx_mean.dim() == 5 and logstd.dim() == 4:
            logstd = logstd.unsqueeze(0).expand(ctx_mean.shape[0], -1, -1, -1, -1)

        mean = ctx_mean
        if shift_vec is not None:
            shift = shift_vec.to(dtype=mean.dtype, device=mean.device)
            while shift.dim() < mean.dim():
                shift = shift.unsqueeze(1)
            mean = mean + shift
        if img_emb is not None and self.bayes_condition_on_image:
            shift = cond_proj(img_emb).to(dtype=mean.dtype, device=mean.device)
            while shift.dim() < mean.dim():
                shift = shift.unsqueeze(1)
            mean = mean + shift

        std = torch.exp(logstd)
        eps = torch.randn_like(mean)
        ctx = mean + std * eps

        if self.bayes_flow_type == "planar":
            if img_emb is None:
                # no conditioning available; use zero params => identity
                return ctx, 0.5 * (mean.pow(2) + std.pow(2) - 1.0 - 2.0 * logstd).mean()

            B = img_emb.shape[0]
            D = img_emb.shape[-1]
            u = self.amor_u(img_emb).view(B, self.bayes_flow_steps, D).unsqueeze(-1)  # (B,K,D,1)
            w = self.amor_w(img_emb).view(B, self.bayes_flow_steps, D).unsqueeze(2)   # (B,K,1,D)
            b = self.amor_b(img_emb).view(B, self.bayes_flow_steps, 1, 1)             # (B,K,1,1)

            # ctx: (B, ..., D) -> (B, M, D)
            orig_shape = ctx.shape
            ctx2 = ctx.view(B, -1, D)
            M = ctx2.shape[1]
            z = ctx2.reshape(B * M, D)

            for k, flow in enumerate(self.planar_flows):
                u_k = u[:, k, :, :].repeat_interleave(M, dim=0)  # (B*M,D,1)
                w_k = w[:, k, :, :].repeat_interleave(M, dim=0)  # (B*M,1,D)
                b_k = b[:, k, :, :].repeat_interleave(M, dim=0)  # (B*M,1,1)
                z, _ = flow(z, u_k, w_k, b_k)

            ctx = z.view(B, M, D).view(orig_shape)
        else:
            for layer in getattr(self, "prompt_flow", []):
                ctx = ctx + torch.tanh(layer(ctx))

        # Residual-gated update: ctx = mean + alpha * (ctx - mean)
        if alpha_param is not None and bool(getattr(self, "bayes_use_residual", False)):
            alpha = torch.clamp(alpha_param.to(dtype=mean.dtype, device=mean.device), 0.0, 1.0)
            ctx = mean + alpha * (ctx - mean)

        # KL(N(mean, std) || N(0, 1))
        kl = 0.5 * (mean.pow(2) + std.pow(2) - 1.0 - 2.0 * logstd).mean()
        return ctx, kl


    #2    
    def _pad_and_concatenate_suffix(ctx, selected_embeddings, tokenized_prompts, prefix, suffix, device):
        ctx = torch.cat([ctx.to(device), selected_embeddings.to(device)], dim=3)
        suffix = suffix[:, :, :, :-selected_embeddings.shape[-2], :]
        insert_idx = prefix.shape[-2] + ctx.shape[-2]
        tokenized_prompts = torch.cat(
            [
                tokenized_prompts[:, :, :, :insert_idx].to(device),
                tokenized_prompts[:, :, :, insert_idx].unsqueeze(-1).repeat(1, 1, 1, selected_embeddings.shape[-2]).to(device),
                tokenized_prompts[:, :, :, insert_idx:].to(device)
            ],
            dim=-1
        )
        tokenized_prompts = tokenized_prompts[:, :, :, :-1]
        return ctx, suffix, tokenized_prompts
    #3
    def _pad_and_concatenate_prefix(prefix, selected_embeddings, tokenized_prompts, suffix, device):
        prefix = torch.cat([prefix.to(device), selected_embeddings.to(device)], dim=3)
        suffix = suffix[:, :, :, :-selected_embeddings.shape[-2], :]
        insert_idx = prefix.shape[-2]
        tokenized_prompts = torch.cat(
            [
                tokenized_prompts[:, :, :, :insert_idx].to(device),
                tokenized_prompts[:, :, :, insert_idx].unsqueeze(-1).repeat(1, 1, 1, selected_embeddings.shape[-2]).to(device),
                tokenized_prompts[:, :, :, insert_idx:].to(device)
            ],
            dim=-1
        )
        tokenized_prompts = tokenized_prompts[:, :, :, :-1]
        return prefix, suffix, tokenized_prompts



    #4 是否使用图像特征
    def _forward(self, img_emb=None):   # Add noise to img_emb   
        force_train_with_img_cls = self.use_bayes_prompt and self.bayes_condition_on_image and img_emb is not None
        if force_train_with_img_cls:
            train_with_img_cls = True
        elif self.train_with_img_cls_prob != 1:
            train_with_img_cls = torch.rand(1).item() <= self.train_with_img_cls_prob
        else:
            train_with_img_cls = True
            

        # 4.0 准备张量（广播到 batch 维度）
        batch_size = img_emb.shape[0] if not img_emb is None else None
        if train_with_img_cls:   
            assert batch_size == img_emb.shape[0]
            # Replicate other tensors if necessary
            prefix_pos = self.token_prefix_pos.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
            suffix_pos = self.token_suffix_pos.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
            prefix_neg = self.token_prefix_neg.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
            suffix_neg = self.token_suffix_neg.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
            # Reshape tokenized prompts to match batch size
            tokenized_prompts_pos = self.tokenized_prompts_pos.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [batch_size, 1, 1, 77]
            tokenized_prompts_neg = self.tokenized_prompts_neg.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [batch_size, 1, 1, 77]
            dim = 3
            # Replicate ctx_pos and ctx_neg to match batch size
            ctx_pos = self.ctx_pos.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)  # [batch_size, 1, 1, 12or6, 768]
            ctx_neg = self.ctx_neg.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)  # [batch_size, 1, 1, 12or6, 768]

            if self.use_bayes_prompt:
                shift_ctx, shift_state = self._maybe_compute_pfl_biases(img_emb)
                if shift_ctx is not None and shift_state is not None:
                    shift_n, shift_a = shift_state
                    shift_pos = shift_ctx + shift_n.expand_as(shift_ctx)
                    shift_neg = shift_ctx + shift_a.expand_as(shift_ctx)
                    ctx_pos, kl_pos = self._bayes_sample_ctx(
                        ctx_pos,
                        self.ctx_pos_logstd,
                        img_emb,
                        self.ctx_cond_pos,
                        shift_vec=shift_pos,
                        alpha_param=self.bayes_residual_alpha_pos if self.bayes_use_residual else None,
                    )
                    ctx_neg, kl_neg = self._bayes_sample_ctx(
                        ctx_neg,
                        self.ctx_neg_logstd,
                        img_emb,
                        self.ctx_cond_neg,
                        shift_vec=shift_neg,
                        alpha_param=self.bayes_residual_alpha_neg if self.bayes_use_residual else None,
                    )
                else:
                    ctx_pos, kl_pos = self._bayes_sample_ctx(
                        ctx_pos,
                        self.ctx_pos_logstd,
                        img_emb,
                        self.ctx_cond_pos,
                        alpha_param=self.bayes_residual_alpha_pos if self.bayes_use_residual else None,
                    )
                    ctx_neg, kl_neg = self._bayes_sample_ctx(
                        ctx_neg,
                        self.ctx_neg_logstd,
                        img_emb,
                        self.ctx_cond_neg,
                        alpha_param=self.bayes_residual_alpha_neg if self.bayes_use_residual else None,
                    )
                self._last_kl = kl_pos + kl_neg
            else:
                self._last_kl = None

            img_emb_pos = img_emb.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, self.normal_num, 1, 1)
            img_emb_neg = img_emb.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, self.anormaly_num, 1, 1)
            # 4.1 用 img_emb 替换 ctx 的第一个 token
            if self.train_with_img_cls_type == 'replace_prefix':
                ctx_pos = torch.cat([img_emb_pos.to(ctx_pos.device), ctx_pos[:, :, :, 1:, :].to(ctx_pos.device)], dim=3)
                ctx_neg = torch.cat([img_emb_neg.to(ctx_neg.device), ctx_neg[:, :, :, 1:, :].to(ctx_neg.device)], dim=3)
            # 4.2 用 img_emb 替换 ctx 的最后一个 token
            elif self.train_with_img_cls_type == 'replace_suffix':
                ctx_pos = torch.cat([ctx_pos[:, :, :, :-1, :].to(ctx_pos.device), img_emb_pos.to(ctx_pos.device)], dim=3)
                ctx_neg = torch.cat([ctx_neg[:, :, :, :-1, :].to(ctx_neg.device), img_emb_neg.to(ctx_neg.device)], dim=3)
            # 4.3 在 prefix 后插入 img_emb，扩展序列长度
            elif self.train_with_img_cls_type == 'pad_prefix':
                prefix_pos, suffix_pos, tokenized_prompts_pos = PromptLearner._pad_and_concatenate_prefix(
                    prefix_pos, img_emb_pos, tokenized_prompts_pos, suffix_pos, prefix_pos.device
                )
                prefix_neg, suffix_neg, tokenized_prompts_neg = PromptLearner._pad_and_concatenate_prefix(
                    prefix_neg, img_emb_neg, tokenized_prompts_neg, suffix_neg, prefix_neg.device
                )
            # 4.4 在 ctx 后插入 img_emb，扩展序列长度    
            elif self.train_with_img_cls_type == 'pad_suffix':
                ctx_pos, suffix_pos, tokenized_prompts_pos = PromptLearner._pad_and_concatenate_suffix(
                    ctx_pos, img_emb_pos, tokenized_prompts_pos, prefix_pos, suffix_pos, ctx_pos.device
                )
                ctx_neg, suffix_neg, tokenized_prompts_neg = PromptLearner._pad_and_concatenate_suffix(
                    ctx_neg, img_emb_neg, tokenized_prompts_neg, prefix_neg, suffix_neg, ctx_neg.device
                )
            
        # 4.5 不使用图像特征    
        else:
            ctx_pos = self.ctx_pos # [1, 1, n_bat, 768]
            ctx_neg = self.ctx_neg

            if self.use_bayes_prompt:
                ctx_pos, kl_pos = self._bayes_sample_ctx(
                    ctx_pos,
                    self.ctx_pos_logstd,
                    None,
                    self.ctx_cond_pos,
                    alpha_param=self.bayes_residual_alpha_pos if self.bayes_use_residual else None,
                )
                ctx_neg, kl_neg = self._bayes_sample_ctx(
                    ctx_neg,
                    self.ctx_neg_logstd,
                    None,
                    self.ctx_cond_neg,
                    alpha_param=self.bayes_residual_alpha_neg if self.bayes_use_residual else None,
                )
                self._last_kl = kl_pos + kl_neg
            else:
                self._last_kl = None
            
            prefix_pos, suffix_pos = self.token_prefix_pos, self.token_suffix_pos
            prefix_neg, suffix_neg = self.token_prefix_neg, self.token_suffix_neg
            tokenized_prompts_pos, tokenized_prompts_neg = self.tokenized_prompts_pos, self.tokenized_prompts_neg
          
            dim = 2


        # 4.6 拼接
        prompts_pos = torch.cat(
            [
                # N(the number of template), 1, dim
                prefix_pos,  # (n_cls, 1, dim)
                ctx_pos,  # (n_cls, n_ctx, dim)
                suffix_pos,  # (n_cls, *, dim)
            ],
            dim=dim,
        )
        prompts_neg = torch.cat(
            [
                prefix_neg,  # (n_cls, 1, dim)
                ctx_neg,  # (n_cls, n_ctx, dim)
                suffix_neg,  # (n_cls, *, dim)
            ],
            dim=dim,
        )

        l, d = prompts_pos.shape[-2:] # (8, 1, 2, 77, 768)
        prompts_pos = prompts_pos.reshape(-1, l, d)
        l, d = prompts_neg.shape[-2:] # (8, 1, 2, 77, 768)
        prompts_neg = prompts_neg.reshape(-1, l, d)
        
        l, d = tokenized_prompts_pos.shape[-2:]
        tokenized_prompts_pos = tokenized_prompts_pos.reshape(-1, d)
        l, d = tokenized_prompts_neg.shape[-2:]
        tokenized_prompts_neg = tokenized_prompts_neg.reshape(-1, d)
        
        prompts = [prompts_pos, prompts_neg]
        tokenized_prompts = [tokenized_prompts_pos, tokenized_prompts_neg]

        if not train_with_img_cls:
            prompts = torch.cat(prompts, dim=0)
            tokenized_prompts = torch.cat(tokenized_prompts, dim=0)

        # 四元组[提示，tokenID, 深层 prompt, 是否使用了图像特征 ]
        return prompts, tokenized_prompts, self.compound_prompts_text, train_with_img_cls


    #5 缓存机制
    def forward(self, img_emb=None):
        if self.use_bayes_prompt:
            # Bayesian mode is stochastic; never cache.
            self.constructed_prompts = self._forward(img_emb)
            return self.constructed_prompts

        if self.train_with_img_cls_type != 'none' or self.train_with_img_cls_prob != 0 or self.constructed_prompts == None:
            self.constructed_prompts = self._forward(img_emb)
        return self.constructed_prompts
