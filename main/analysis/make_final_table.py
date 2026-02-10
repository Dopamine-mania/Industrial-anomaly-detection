import argparse
import csv
from pathlib import Path


def read_mean_row(summary_csv: Path) -> dict:
    with summary_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    mean_row = next((r for r in rows if (r.get("objects") or "").strip().lower() == "mean"), None)
    if mean_row is None:
        raise ValueError(f"No 'mean' row found in {summary_csv}")
    return mean_row


def pick_floats(row: dict, keys: list[str]) -> dict:
    out = {}
    for k in keys:
        v = row.get(k, "")
        try:
            out[k] = float(v)
        except Exception:
            out[k] = None
    return out


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--deliverables_root",
        type=Path,
        default=Path("main/deliverables/ours_best_zs_fixedprompts_dino"),
    )
    ap.add_argument("--out_csv", type=Path, default=Path("main/analysis/final_table.csv"))
    ap.add_argument("--out_md", type=Path, default=Path("main/analysis/final_table.md"))
    args = ap.parse_args()

    metrics = ["pixel_auroc", "pixel_ap", "pixel_aupro", "image_auroc", "image_ap", "image_f1"]
    entries = [
        ("MVTec (Best, fixed prompts + DINOv2)", args.deliverables_root / "mvtec" / "summary.csv"),
        ("MVTec (Bayes refined, MC=8, sigma=0.001)", args.deliverables_root / "mvtec_bayes_refined" / "summary.csv"),
        ("VisA (Best, fixed prompts + DINOv2, map_max)", args.deliverables_root / "visa" / "summary.csv"),
        ("VisA (Bayes refined, MC=8, sigma=0.001, map_max)", args.deliverables_root / "visa_bayes_refined" / "summary.csv"),
    ]

    rows = []
    for name, path in entries:
        mean_row = read_mean_row(path)
        row = {"setting": name}
        row.update(pick_floats(mean_row, metrics))
        rows.append(row)

    fieldnames = ["setting", *metrics]
    write_csv(args.out_csv, rows, fieldnames)

    try:
        from tabulate import tabulate

        md = tabulate(
            [[r.get(k, "") for k in fieldnames] for r in rows],
            headers=fieldnames,
            tablefmt="github",
            floatfmt=".4f",
        )
    except Exception:
        md_lines = ["| " + " | ".join(fieldnames) + " |", "| " + " | ".join(["---"] * len(fieldnames)) + " |"]
        for r in rows:
            md_lines.append("| " + " | ".join(str(r.get(k, "")) for k in fieldnames) + " |")
        md = "\n".join(md_lines)

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md + "\n", encoding="utf-8")
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()

