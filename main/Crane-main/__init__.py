import os

# Keep datasets outside the repo by default; override via env var.
DATASETS_ROOT = os.environ.get(
    "CRANE_DATASETS_ROOT",
    os.path.join(os.path.dirname(__file__), "data"),
)
