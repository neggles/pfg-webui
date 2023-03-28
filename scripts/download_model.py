# このコードはhttps://github.com/kohya-ss/sd-scripts/blob/main/finetune/tag_images_by_wd14_tagger.pyを参考にしていますというかパクっています。
from os import PathLike
from pathlib import Path

from huggingface_hub import snapshot_download


TAGGER_REPO = "SmilingWolf/wd-v1-4-vit-tagger-v2"
TAGGER_DIR = "wd-v1-4-vit-tagger-v2"
TAGGER_IGNORE = ["*.onnx", "*.md", ".git*"]

PFG_REPO = "furusu/PFG"
PFG_IGNORE = ["*.md", "*.png", ".git*"]
ONNX_FILE = "wd-v1-4-vit-tagger-v2-last-pooling-layer.onnx"


def download(models_path: PathLike):
    if not isinstance(models_path, Path):
        models_path = Path(models_path)

    tagger_path = models_path.joinpath(TAGGER_DIR)
    if not tagger_path.exists():
        # Get WD14 Tagger in TensorFlow/Keras format
        print(f"Downloading WD1.4 tagger (TensorFlow) from HF repo: {TAGGER_REPO}")
        snapshot_download(
            repo_id=TAGGER_REPO,
            ignore_patterns=TAGGER_IGNORE,
            local_dir=tagger_path,
        )
    else:
        print(f"WD1.4 tagger found at {tagger_path}.")

    print(f"Updating ONNX tagger and model tensors from HF repo: {PFG_REPO}")
    snapshot_download(
        repo_id=PFG_REPO,
        ignore_patterns=PFG_IGNORE,
        local_dir=models_path,
    )


__all__ = [
    "download",
    "TAGGER_DIR",
    "TAGGER_REPO",
    "PFG_REPO",
    "ONNX_FILE",
]
