import argparse
from pathlib import Path
import torch

from opticsim.fields import make_grid, plane_wave, single_slit_aperture, apply_aperture
from opticsim.asm import asm_transfer_function, asm_propagate
from opticsim.io_utils import ensure_dir, save_intensity_png, save_config, print_env_info


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args():
    p = argparse.ArgumentParser(description="Week 1 - PyTorch ASM single-slit simulator (Apple Silicon MPS supported)")
    p.add_argument("--N", type=int, default=1024, help="Grid size (N x N)")
    p.add_argument("--dx", type=float, default=10e-6, help="Pixel pitch in meters")
    p.add_argument("--wavelength", type=float, default=633e-9, help="Wavelength in meters")
    p.add_argument("--z", type=float, default=0.10, help="Propagation distance in meters")
    p.add_argument("--slit_width", type=float, default=100e-6, help="Single-slit width in meters")
    p.add_argument("--evanescent", type=str, default="keep", choices=["keep", "cut"],
                   help="How to handle evanescent components (keep or cut)")
    p.add_argument("--outdir", type=str, default="results/run1", help="Output directory")
    p.add_argument("--dtype", type=str, default="complex64", choices=["complex64", "complex128"],
                   help="Complex dtype (recommend complex64 for MPS)")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()

    # MPS is most reliable with complex64
    dtype = torch.complex64 if args.dtype == "complex64" else torch.complex128
    if device.type == "mps" and dtype == torch.complex128:
        print("[WARN] MPS backend may be unstable with complex128. Consider using --dtype complex64.")

    outdir = Path(args.outdir)
    ensure_dir(outdir)
    ensure_dir(outdir / "results")

    print_env_info(device)

    # 1) build input field U0(x,y,0)
    X, Y = make_grid(args.N, args.dx, device=device)
    U0 = plane_wave(args.N, device=device, dtype=dtype)

    A = single_slit_aperture(X, slit_width=args.slit_width)
    U0 = apply_aperture(U0, A)

    # 2) build transfer function H(kx,ky)
    H = asm_transfer_function(
        N=args.N,
        dx=args.dx,
        wavelength=args.wavelength,
        z=args.z,
        device=device,
        dtype=dtype,
        evanescent=args.evanescent,
    )

    # 3) propagate
    Uz = asm_propagate(U0, H)

    # 4) intensity
    I = torch.abs(Uz) ** 2

    # 5) save outputs (always save on CPU)
    intensity_path = outdir / "intensity.png"
    config_path = outdir / "config.json"

    save_intensity_png(I, intensity_path, title="ASM single-slit intensity")
    save_config({
        **vars(args),
        "device": str(device),
        "torch_version": torch.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
    }, config_path)

    print(f"[OK] saved: {intensity_path}")
    print(f"[OK] saved: {config_path}")


if __name__ == "__main__":
    main()
