import argparse
import shutil
import zipfile
from pathlib import Path


def copytree(src: Path, dst: Path, ignore_names: set[str]):
    def _ignore(_dir: str, names: list[str]):
        return [n for n in names if n in ignore_names]

    shutil.copytree(src, dst, ignore=_ignore, dirs_exist_ok=True)


def zip_dir(src_dir: Path, out_zip: Path):
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in src_dir.rglob("*"):
            if p.is_dir():
                continue
            arc = p.relative_to(src_dir)
            zf.write(p, arcname=str(arc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", type=Path, default=Path("."))
    ap.add_argument("--out_dir", type=Path, default=Path("main/deliverables/delivery_package"))
    ap.add_argument("--out_zip", type=Path, default=Path("main/deliverables/delivery_package.zip"))
    args = ap.parse_args()

    repo_root = args.repo_root.resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    out_zip = (repo_root / args.out_zip).resolve()

    if out_dir.exists():
        shutil.rmtree(out_dir)

    (out_dir / "code").mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "results").mkdir(parents=True, exist_ok=True)
    (out_dir / "heatmaps").mkdir(parents=True, exist_ok=True)

    # 1) code/
    crane_src = repo_root / "main/Crane-main"
    crane_dst = out_dir / "code/Crane-main"
    copytree(
        crane_src,
        crane_dst,
        ignore_names={
            "results",
            "vis_img",
            "checkpoints",
            "__pycache__",
            ".pytest_cache",
            ".ipynb_checkpoints",
        },
    )

    analysis_src = repo_root / "main/analysis"
    analysis_dst = out_dir / "code/analysis"
    analysis_dst.mkdir(parents=True, exist_ok=True)
    for name in [
        "make_final_table.py",
        "plot_sota_comparison.py",
        "bayes_variance_analysis.py",
        "collect_results.py",
        "plot_ablation.py",
    ]:
        p = analysis_src / name
        if p.exists():
            shutil.copy2(p, analysis_dst / name)

    # 2) checkpoints/
    # Our best setting is training-free; provide a transparent "checkpoint" manifest as a .pth.
    try:
        import torch
        import subprocess

        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True).strip()
        )
        ckpt = {
            "name": "best_model_mvtec (zero-shot/training-free)",
            "note": "This project’s best configuration is zero-shot (no fine-tuned weights). "
            "Run with --skip_checkpoint_load True. This file stores only configuration/provenance.",
            "commit": commit,
            "recommended_test_args": {
                "dataset": "mvtec",
                "dino_model": "dinov2",
                "fixed_prompts": True,
                "fixed_prompts_reduce": "mean",
                "fixed_prompts_bayes": True,
                "fixed_prompts_bayes_num_samples": 8,
                "fixed_prompts_bayes_init_logstd": -6.907755278982137,  # sigma=0.001
                "use_resclip": True,
                "resclip_alpha": 0.5,
                "sigma": 4,
            },
        }
        torch.save(ckpt, out_dir / "checkpoints/best_model_mvtec.pth")
    except Exception:
        # If torch is unavailable, fall back to a plain text note.
        (out_dir / "checkpoints/best_model_mvtec.pth").write_text(
            "Zero-shot/training-free configuration; no fine-tuned weights.\n", encoding="utf-8"
        )

    # 3) results/
    shutil.copy2(repo_root / "main/analysis/final_table.csv", out_dir / "results/final_table.csv")
    shutil.copy2(repo_root / "main/analysis/sota_comparison.png", out_dir / "results/sota_comparison.png")
    shutil.copy2(
        repo_root / "main/analysis/bayes_variance/mvtec_bottle_mc_std_hist.png",
        out_dir / "results/variance_analysis.png",
    )

    # Include ablation plots as extra evidence (optional but helpful).
    for p in [
        repo_root / "main/analysis/ablation_pixel_auroc_mvtec.png",
        repo_root / "main/analysis/ablation_image_auroc_mvtec.png",
    ]:
        if p.exists():
            shutil.copy2(p, out_dir / "results" / p.name)

    # 4) heatmaps/
    heat_src = repo_root / "main/deliverables/heatmaps"
    if heat_src.exists():
        copytree(heat_src, out_dir / "heatmaps", ignore_names={"__pycache__"})

    # README + report are written outside of this script (to keep the script small and stable).
    # The caller is expected to place README_Customer.md and TECHNICAL_REPORT.md under out_dir.

    # 5) zip
    zip_dir(out_dir, out_zip)
    print(f"Wrote folder: {out_dir}")
    print(f"Wrote zip:    {out_zip}")


if __name__ == "__main__":
    main()

