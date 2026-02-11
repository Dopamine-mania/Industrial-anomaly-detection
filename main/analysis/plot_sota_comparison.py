import argparse
import csv
from pathlib import Path


def read_mean_pixel_auroc(summary_csv: Path) -> float:
    with summary_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    mean = next(r for r in rows if (r.get("objects") or "").strip().lower() == "mean")
    return float(mean["pixel_auroc"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_png", type=Path, default=Path("main/analysis/sota_comparison.png"))
    ap.add_argument(
        "--ours_summary_csv",
        type=Path,
        default=Path("main/deliverables/ours_best_zs_fixedprompts_dino/mvtec/summary.csv"),
    )
    ap.add_argument(
        "--base_summary_csv",
        type=Path,
        default=Path("main/Crane-main/results/baseline_clip_fixedprompts_mvtec/test_on_mvtec/0d4ee9eb/epoch_0/summary.csv"),
    )
    ap.add_argument("--crane_pdf_pixel", type=float, default=0.912, help="Crane Pixel AUROC from client PDF (fraction)")
    ap.add_argument("--bayes_pfl_pdf_pixel", type=float, default=0.918, help="Bayes-PFL Pixel AUROC from client PDF (fraction)")
    args = ap.parse_args()

    base = read_mean_pixel_auroc(args.base_summary_csv)
    ours = read_mean_pixel_auroc(args.ours_summary_csv)
    crane = float(args.crane_pdf_pixel)
    bayes = float(args.bayes_pfl_pdf_pixel)

    labels = [
        "Baseline\n(Modified WinCLIP)",
        "Crane\n(PDF)",
        "Bayes-PFL\n(PDF)",
        "Ours\n(Zero-shot)",
    ]
    values = [base * 100, crane * 100, bayes * 100, ours * 100]
    colors = ["#9aa0a6", "#4e79a7", "#59a14f", "#f28e2b"]

    import matplotlib.pyplot as plt

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.5, 4))
    bars = plt.bar(labels, values, color=colors)
    plt.ylabel("Pixel AUROC (%)")
    plt.ylim(80, 95)
    plt.title("MVTec AD Pixel AUROC — SOTA Comparison")
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.2, f"{v:.1f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(args.out_png, dpi=220)
    print(f"Wrote {args.out_png}")


if __name__ == "__main__":
    main()

