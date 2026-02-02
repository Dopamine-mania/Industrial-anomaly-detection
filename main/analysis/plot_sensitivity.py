import argparse
import csv
import re
from pathlib import Path


def read_csv(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(x: str):
    try:
        return float(x)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", type=Path, default=Path("main/analysis/summary.csv"))
    ap.add_argument("--out_png", type=Path, default=Path("main/analysis/sensitivity.png"))
    ap.add_argument("--metric", type=str, default="image_auroc", choices=["image_auroc", "pixel_auroc"])
    ap.add_argument("--x_key", type=str, default="R", choices=["R", "K"])
    ap.add_argument("--tag_regex", type=str, default=r".*R(?P<R>\\d+).*", help="regex extracting x from model_name")
    args = ap.parse_args()

    rows = read_csv(args.summary_csv)
    pat = re.compile(args.tag_regex)
    points = {}
    for r in rows:
        model_name = r.get("model_name", "")
        m = pat.match(model_name)
        if not m:
            continue
        x = int(m.group(args.x_key))
        v = to_float(r.get(args.metric, ""))
        if v is None:
            continue
        points.setdefault(x, []).append(v)

    if not points:
        raise SystemExit("No matching rows; adjust --tag_regex or model_name tags.")

    xs = sorted(points.keys())
    ys = [sum(points[x]) / len(points[x]) for x in xs]

    import matplotlib.pyplot as plt

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(xs, ys, marker="o")
    plt.xlabel(args.x_key)
    plt.ylabel(args.metric)
    plt.title(f"Sensitivity ({args.metric}) vs {args.x_key}")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=200)
    print(f"Wrote {args.out_png}")


if __name__ == "__main__":
    main()

