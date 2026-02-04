import argparse
import csv
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path):
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def to_float(x: str):
    try:
        return float(x)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", type=Path, default=Path("main/analysis/summary.csv"))
    ap.add_argument("--out_png", type=Path, default=None)
    ap.add_argument("--metric", type=str, default="image_auroc", choices=["image_auroc", "pixel_auroc"])
    ap.add_argument(
        "--order",
        nargs="+",
        default=[
            "trained_on_mvtec_baseline",
            "trained_on_mvtec_bayes_dino",
            "trained_on_visa_baseline",
            "trained_on_visa_bayes_dino",
        ],
        help="model_name order for plotting",
    )
    args = ap.parse_args()

    rows = read_csv(args.summary_csv)
    grouped = defaultdict(list)
    for r in rows:
        model_name = r.get("model_name", "")
        v = to_float(r.get(args.metric, ""))
        if v is None:
            continue
        grouped[model_name].append(v)

    labels = []
    values = []
    for m in args.order:
        if m not in grouped:
            continue
        labels.append(m.replace("trained_on_mvtec_", "").replace("trained_on_visa_", ""))
        values.append(sum(grouped[m]) / len(grouped[m]))

    if not values:
        raise SystemExit("No matching rows for plotting; check --order and --summary_csv")

    import matplotlib.pyplot as plt

    out_png = args.out_png or Path(f"main/analysis/ablation_{args.metric}.png")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.bar(labels, values)
    plt.ylabel(args.metric)
    plt.title(f"Ablation ({args.metric})")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
