import argparse
from pathlib import Path


def parse_markdown_table_mean(log_txt: Path) -> dict | None:
    lines = log_txt.read_text(encoding="utf-8", errors="replace").splitlines()
    table_lines = [ln for ln in lines if ln.strip().startswith("|")]
    if not table_lines:
        return None

    # locate header
    header_idx = None
    for i, ln in enumerate(table_lines):
        if "objects" in ln and "pixel_auroc" in ln:
            header_idx = i
            break
    if header_idx is None or header_idx + 2 >= len(table_lines):
        return None

    header = [c.strip() for c in table_lines[header_idx].strip().strip("|").split("|")]
    rows = []
    for ln in table_lines[header_idx + 2 :]:
        if not ln.strip().startswith("|"):
            break
        cols = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(cols) != len(header):
            continue
        row = dict(zip(header, cols))
        rows.append(row)

    mean_row = next((r for r in rows if r.get("objects") == "mean"), None)
    if mean_row is None:
        return None

    out = {}
    for k, v in mean_row.items():
        if k == "objects":
            continue
        try:
            out[k] = float(v)
        except ValueError:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", type=Path, default=Path("main/Crane-main/results"))
    ap.add_argument("--out_csv", type=Path, default=Path("main/analysis/summary.csv"))
    args = ap.parse_args()

    rows = []
    for log_txt in args.results_root.rglob("epoch_*/log.txt"):
        parts = log_txt.parts
        # .../results/<model_name>/test_on_<dataset>/<log_dir>/epoch_x/log.txt
        try:
            results_idx = parts.index("results")
            model_name = parts[results_idx + 1]
            test_on = parts[results_idx + 2]  # test_on_<dataset>
            test_dataset = test_on.replace("test_on_", "")
        except Exception:
            continue

        metrics = parse_markdown_table_mean(log_txt)
        if metrics is None:
            continue

        rows.append(
            {
                "model_name": model_name,
                "test_dataset": test_dataset,
                "log_path": str(log_txt),
                **metrics,
            }
        )

    if not rows:
        raise SystemExit(f"No results found under {args.results_root}")

    # write simple CSV (no pandas dependency)
    keys = sorted({k for r in rows for k in r.keys()})
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")

    print(f"Wrote {args.out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

