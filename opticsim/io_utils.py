import json
from pathlib import Path
import torch
import matplotlib.pyplot as plt


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_intensity_png(intensity: torch.Tensor, out_path: Path, title: str = ""):
    # Always move to CPU for plotting
    I = intensity.detach().cpu().float().numpy()
    plt.figure()
    plt.imshow(I, cmap="gray")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_config(config: dict, out_path: Path):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def print_env_info(device: torch.device):
    print("=== Environment ===")
    print(f"torch: {torch.__version__}")
    print(f"device: {device}")
    print(f"mps_available: {torch.backends.mps.is_available()}")
    print("===================")
