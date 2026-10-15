"""Wan 2.2 model files: which ones each mode needs, and downloading them.

The weights (~69 GB for both modes) are too big to bake into a Runpod image, so the
worker downloads them on startup. With a Runpod network volume attached they are
stored there once and reused by every worker; without one, each new worker
downloads them to its container disk.
"""

import os
import shutil
from pathlib import Path

# Comfy-Org's repackaged Wan 2.2 files, pinned to a commit for reproducibility.
REPO_ID = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
REVISION = "ee6f4a40737a995bf5818954cfce6d59443b0f04"

SHARED_FILES = (
    "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    "split_files/vae/wan_2.1_vae.safetensors",
)

MODE_FILES = {
    "t2v": (
        "split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
        "split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
        "split_files/loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors",
        "split_files/loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors",
    ),
    "i2v": (
        "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
        "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
        "split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
        "split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
    ),
}

# ComfyUI model folders, relative to <models_dir>/split_files.
COMFY_FOLDERS = ("diffusion_models", "loras", "text_encoders", "vae")


def enabled_modes() -> tuple[str, ...]:
    """Modes to serve, from WAN_MODES (default: both)."""
    modes = tuple(
        m.strip() for m in os.environ.get("WAN_MODES", "t2v,i2v").split(",") if m.strip()
    )
    unknown = [m for m in modes if m not in MODE_FILES]
    if not modes or unknown:
        msg = f"WAN_MODES must be a comma-separated subset of {list(MODE_FILES)}, got {modes}"
        raise ValueError(msg)
    return modes


def models_dir() -> Path:
    """Network volume if attached (shared, persistent), else the container disk."""
    if "WAN_MODELS_DIR" in os.environ:
        return Path(os.environ["WAN_MODELS_DIR"])
    if Path("/runpod-volume").is_dir():
        return Path("/runpod-volume/wan-models")
    return Path("/ComfyUI/models/wan")


def files_for(modes: tuple[str, ...]) -> tuple[str, ...]:
    return SHARED_FILES + tuple(f for m in modes for f in MODE_FILES[m])


def missing_files(modes: tuple[str, ...], target: Path) -> list[str]:
    return [f for f in files_for(modes) if not (target / f).is_file()]


def check_free_space(needed: int, target: Path) -> None:
    """Fail with a clear message instead of a half-finished download."""
    target.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(target).free
    margin = 2 * 1024**3
    if free < needed + margin:
        msg = (
            f"Not enough disk space in {target}: the models need {needed / 1e9:.0f} GB, "
            f"{free / 1e9:.0f} GB is free. Attach a network volume of at least 100 GB, "
            "raise the container disk, or set WAN_MODES=t2v (or i2v) to download one mode."
        )
        raise RuntimeError(msg)


def ensure_models(modes: tuple[str, ...], target: Path) -> None:
    """Download any missing model files into ``target`` (resumable, lock-protected)."""
    missing = missing_files(modes, target)
    if not missing:
        print(f"Models ready in {target}", flush=True)
        return

    # Keep Hugging Face's own caches off the (small) container disk: write
    # straight into ``target`` and don't keep a second copy in the Xet chunk cache.
    os.environ.setdefault("HF_HOME", str(target / ".hf-home"))
    os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")
    from huggingface_hub import HfApi, hf_hub_download

    sizes = HfApi().get_paths_info(REPO_ID, missing, revision=REVISION)
    check_free_space(sum(getattr(f, "size", 0) or 0 for f in sizes), target)

    for filename in missing:
        print(f"Downloading {filename} ...", flush=True)
        hf_hub_download(REPO_ID, filename, revision=REVISION, local_dir=target)
    print(f"Models ready in {target}", flush=True)


def write_extra_model_paths(target: Path, config_path: Path) -> Path:
    """Point ComfyUI's model folders at the downloaded files."""
    lines = ["wan:", f"  base_path: {target / 'split_files'}"]
    lines += [f"  {folder}: {folder}" for folder in COMFY_FOLDERS]
    config_path.write_text("\n".join(lines) + "\n")
    return config_path
