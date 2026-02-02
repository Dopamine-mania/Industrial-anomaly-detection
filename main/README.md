# Industrial Anomaly Detection (Main Workspace)

本目录是**唯一**允许改动/产出结果的工作区，用于把 `客户需求/Crane-main` 复制出来后进行二次开发，避免污染客户原始材料，方便回溯。

## 目录结构

- `main/Crane-main/`：主战场（Crane 代码底座 + Bayes-PFL 文本端插件 + feature_refinement_module 开关）
- `main/experiments/`：一键训练/评测脚本（MVTec ↔ VisA）
- `main/analysis/`：结果汇总与画图脚本（消融柱状图、超参敏感度曲线）

## 数据路径（强制外置，防止误写入）

代码默认读取 `CRANE_DATASETS_ROOT`：

```bash
export CRANE_DATASETS_ROOT=/path/to/datasets
```

数据集目录期望形如：

```text
$CRANE_DATASETS_ROOT/
  mvtec/
    meta.json
    ...
  visa/
    meta.json
    ...
```

训练严格读取 `train` split（并强制 normal-only）；评测读取 `test` split。

## 核心开关

- Bayes-PFL 文本端插件：`--use_bayes_prompt True`
  - 采样次数：`--bayes_num_samples R`
  - flow 步数：`--bayes_flow_steps K`
  - 是否用图像特征做分布条件：`--bayes_condition_on_image True|False`
- Attention 加分项（最小风险复用）：`--use_feature_refinement_module True`
  - `--frm_type scalar|linear`

## 一键实验（MVTec ↔ VisA）

```bash
export CRANE_DATASETS_ROOT=/path/to/datasets
bash main/experiments/run_ablation.sh 0
```

## 汇总与画图

```bash
python3 main/analysis/collect_results.py
python3 main/analysis/plot_ablation.py --metric image_auroc
python3 main/analysis/plot_ablation.py --metric pixel_auroc
```
