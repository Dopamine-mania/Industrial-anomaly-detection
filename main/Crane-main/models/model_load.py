import hashlib
import os
import urllib
import warnings
from typing import Union, List
from pkg_resources import packaging
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from tqdm import tqdm
import numpy as np
from .build_model import build_model
from torchvision.transforms import InterpolationMode

if packaging.version.parse(torch.__version__) < packaging.version.parse("1.7.1"):
    warnings.warn("PyTorch version 1.7.1 or higher is recommended")
# https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/pretrained.py#L343
__all__ = ["available_models", "load"]
_MODELS = {
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}



# 1 模型自动下载
def _download( url: str, cache_dir: Union[str, None] = None,):
    # 1.1 缓存目录创建文件夹
    if not cache_dir:
        cache_dir = os.path.expanduser("~/.cache/clip")
    os.makedirs(cache_dir, exist_ok=True)
    # 1.2 提取文件名和预期 SHA256 校验码
    filename = os.path.basename(url)
    if 'openaipublic' in url:
        expected_sha256 = url.split("/")[-2]
    elif 'mlfoundations' in url:
        expected_sha256 = os.path.splitext(filename)[0].split("-")[-1]
    else:
        expected_sha256 = ''
    # 1.3 构造本地保存路径，目标路径是否合法
    download_target = os.path.join(cache_dir, filename)
    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")
    # 1.4  如果文件已存在 → 尝试复用
    if os.path.isfile(download_target):
        if expected_sha256:
            if hashlib.sha256(open(download_target, "rb").read()).hexdigest().startswith(expected_sha256):
                return download_target
            else:
                warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")
        else:
            return download_target
    # 1.5 开始下载（带进度条）
    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.headers.get("Content-Length")), ncols=80, unit='iB', unit_scale=True) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break
                output.write(buffer)
                loop.update(len(buffer))
    # 1.6 下载后再次校验
    if expected_sha256 and not hashlib.sha256(open(download_target, "rb").read()).hexdigest().startswith(expected_sha256):
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")
    # 1.7 返回本地路径
    return download_target



#2 RGB图像
def _convert_image_to_rgb(image):
    return image.convert("RGB")
#3 图像预处理
def _transform(n_px):
    return Compose([
        Resize((n_px, n_px), interpolation=InterpolationMode.BICUBIC),
        #CenterCrop(n_px), # rm center crop to explain whole image
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


#4 模型列表
def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())

#5 权重加载
def load_state_dict(checkpoint_path: str, map_location='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    if next(iter(state_dict.items()))[0].startswith('module'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict

#6
def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", design_details = None, jit: bool = False, download_root: str = None):
    
    """Load a CLIP model
    -------    
    Parameters
    1.name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict
    2.device : Union[str, torch.device]
        The device to put the loaded model
    3.jit : bool
        Whether to load the optimized JIT model or more hackable non-JIT model (default).
    4.download_root: str
        path to download the model files; by default, it uses "~/.cache/clip"
    -------
    Returns
    1.model : torch.nn.Module
        The CLIP model
    2.preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    # 6.1 确定模型文件路径
    print("name", name)

    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    # 6.2 尝试加载JIT格式模型【一般不用】
    with open(model_path, 'rb') as opened_file:
        try:
            # loading JIT archive
            model = torch.jit.load(opened_file, map_location=device if jit else "cpu").eval()
            state_dict = None
        except RuntimeError:
            # loading saved state dict
            if jit:
                warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
                jit = False
            state_dict = torch.load(opened_file, map_location="cpu")
    # 6.3 非JIT格式
    if not jit:
        model = build_model(name, state_dict or model.state_dict(), design_details).to(device)
        if str(device) == "cpu":
            model.float()
        return model, _transform(model.visual.input_resolution)


    # 6.4 patch the device names （应用设备修复）
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(node["value"]).startswith("cuda"):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # 6.4 patch dtype to float32 on CPU（数据类型修复）
    if str(device) == "cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
                        if inputs[i].node()["value"] == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)
        model.float()

    return model, _transform(model.input_resolution.item())