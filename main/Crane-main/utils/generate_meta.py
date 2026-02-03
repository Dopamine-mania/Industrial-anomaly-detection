import argparse
import csv
import json
import os
from pathlib import Path


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def iter_images(dir_path: Path):
    if not dir_path.exists():
        return
    for p in sorted(dir_path.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def find_mvtec_mask(gt_dir: Path, img_path: Path) -> Path | None:
    # Default MVTec naming: <stem>_mask.png
    cand = gt_dir / f"{img_path.stem}_mask.png"
    if cand.exists():
        return cand
    # Fallback: try same suffix
    cand2 = gt_dir / f"{img_path.stem}_mask{img_path.suffix}"
    if cand2.exists():
        return cand2
    # Fallback: any file with same stem prefix
    for p in gt_dir.glob(f"{img_path.stem}_mask.*"):
        if p.is_file():
            return p
    return None


def generate_mvtec_like_meta(dataset_root: Path) -> dict:
    meta = {"train": {}, "test": {}}
    for cls_dir in sorted(dataset_root.iterdir()):
        if not cls_dir.is_dir():
            continue
        train_good = cls_dir / "train" / "good"
        test_dir = cls_dir / "test"
        if not train_good.exists() or not test_dir.exists():
            continue

        cls_name = cls_dir.name
        meta["train"].setdefault(cls_name, [])
        meta["test"].setdefault(cls_name, [])

        # train/good only
        for img in iter_images(train_good):
            meta["train"][cls_name].append(
                {
                    "img_path": str(img.relative_to(dataset_root).as_posix()),
                    "mask_path": "",
                    "cls_name": cls_name,
                    "specie_name": "good",
                    "anomaly": 0,
                }
            )

        # test: good + defects
        for specie_dir in sorted(test_dir.iterdir()):
            if not specie_dir.is_dir():
                continue
            specie = specie_dir.name
            if specie == "good":
                for img in iter_images(specie_dir):
                    meta["test"][cls_name].append(
                        {
                            "img_path": str(img.relative_to(dataset_root).as_posix()),
                            "mask_path": "",
                            "cls_name": cls_name,
                            "specie_name": "good",
                            "anomaly": 0,
                        }
                    )
                continue

            gt_dir = cls_dir / "ground_truth" / specie
            for img in iter_images(specie_dir):
                mask = find_mvtec_mask(gt_dir, img)
                meta["test"][cls_name].append(
                    {
                        "img_path": str(img.relative_to(dataset_root).as_posix()),
                        "mask_path": str(mask.relative_to(dataset_root).as_posix()) if mask else "",
                        "cls_name": cls_name,
                        "specie_name": specie,
                        "anomaly": 1,
                    }
                )
    if not meta["train"] or not meta["test"]:
        raise RuntimeError(
            f"Did not detect MVTec-like structure under {dataset_root}. "
            "Expected <class>/(train/good, test/*, ground_truth/*)."
        )
    return meta


def generate_visa_official_meta(dataset_root: Path, split_csv_name: str = "1cls.csv") -> dict:
    split_csv = dataset_root / "split_csv" / split_csv_name
    if not split_csv.exists():
        raise RuntimeError(f"VisA split file not found: {split_csv}")

    meta = {"train": {}, "test": {}}
    with split_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"object", "split", "label", "image", "mask"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise RuntimeError(f"Unexpected VisA split CSV header: {reader.fieldnames}")

        for row in reader:
            obj = row["object"].strip()
            split = row["split"].strip()
            label = row["label"].strip().lower()
            img_rel = row["image"].strip()
            mask_rel = (row.get("mask") or "").strip()

            if split not in {"train", "test"}:
                continue
            if split == "train":
                # Strictly follow the normal-only training protocol
                if label != "normal":
                    continue
                meta["train"].setdefault(obj, []).append(
                    {
                        "img_path": img_rel,
                        "mask_path": "",
                        "cls_name": obj,
                        "specie_name": "good",
                        "anomaly": 0,
                    }
                )
            else:
                meta["test"].setdefault(obj, [])
                is_anom = 0 if label == "normal" else 1
                meta["test"][obj].append(
                    {
                        "img_path": img_rel,
                        "mask_path": mask_rel if is_anom else "",
                        "cls_name": obj,
                        "specie_name": "good" if is_anom == 0 else label,
                        "anomaly": is_anom,
                    }
                )

    if not meta["train"] or not meta["test"]:
        raise RuntimeError(f"Failed to build VisA meta from {split_csv}")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="Dataset root (e.g. /data/mvtec_ad or /data/visa)")
    ap.add_argument("--out", type=Path, default=None, help="Output meta.json path (default: <root>/meta.json)")
    ap.add_argument("--visa_split_csv", type=str, default="1cls.csv", help="VisA official split CSV under split_csv/")
    args = ap.parse_args()

    dataset_root = args.root.resolve()
    out_path = (args.out or (dataset_root / "meta.json")).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Detect VisA official structure first
    if (dataset_root / "split_csv").exists() and (dataset_root / "split_csv" / args.visa_split_csv).exists():
        meta = generate_visa_official_meta(dataset_root, split_csv_name=args.visa_split_csv)
    else:
        meta = generate_mvtec_like_meta(dataset_root)
    out_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    # quick stats
    n_train = sum(len(v) for v in meta["train"].values())
    n_test = sum(len(v) for v in meta["test"].values())
    print(f"Classes: {len(meta['train'])}, train samples: {n_train}, test samples: {n_test}")


if __name__ == "__main__":
    main()
