from pathlib import Path
import json
import torch
import matplotlib.pyplot as plt

from opticsim.fields_torch import get_device, normalize01, target_from_slit
from opticsim.asm_torch import ASMPropagator


def save_png(t: torch.Tensor, path: Path, title: str):
    img = t.detach().cpu().float().numpy()
    plt.figure()
    plt.imshow(img, cmap="gray")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    device = get_device()
    dtype = torch.complex64  # MPS 안정성 위해 권장

    # ---- parameters ----
    N = 512
    dx = 10e-6
    wavelength = 633e-9
    z = 0.10
    slit_width = 120e-6

    outdir = Path("results/week2_gradcheck")
    outdir.mkdir(parents=True, exist_ok=True)

    print("device:", device)
    print("torch:", torch.__version__)
    print("mps:", torch.backends.mps.is_available())

    # ---- target intensity (toy target) ----
    target_I = target_from_slit(N, dx, device, slit_width)  # float32
    target_I = normalize01(target_I)

    # ---- differentiable propagator ----
    propagator = ASMPropagator(
        N=N, dx=dx, wavelength=wavelength, z=z,
        device=device, dtype=dtype, evanescent="keep"
    ).to(device)

    # ---- learnable phase parameter (no neural net) ----
    # phase is learnable -> U0 = exp(i*phase)
    phase = torch.nn.Parameter(2 * torch.pi * torch.rand((N, N), device=device, dtype=torch.float32))

    # ---- forward + loss ----
    U0 = torch.exp(1j * phase).to(dtype)  # complex field
    Uz = propagator(U0)
    I = torch.abs(Uz) ** 2
    I = normalize01(I)

    loss = torch.mean((I - target_I) ** 2)

    # ---- backward ----
    loss.backward()

    # ---- check gradients ----
    grad_mean = phase.grad.abs().mean().item()
    grad_max = phase.grad.abs().max().item()

    print(f"loss: {loss.item():.6e}")
    print(f"phase.grad | mean: {grad_mean:.6e}, max: {grad_max:.6e}")

    # ---- save artifacts ----
    save_png(target_I, outdir / "target_intensity.png", "Target intensity (toy)")
    save_png(I, outdir / "pred_intensity.png", "Pred intensity (ASM forward)")

    with open(outdir / "run.json", "w", encoding="utf-8") as f:
        json.dump({
            "N": N,
            "dx": dx,
            "wavelength": wavelength,
            "z": z,
            "slit_width": slit_width,
            "device": str(device),
            "dtype": str(dtype),
            "loss": float(loss.item()),
            "grad_mean": float(grad_mean),
            "grad_max": float(grad_max),
        }, f, indent=2)

    print(f"[OK] saved: {outdir}")


if __name__ == "__main__":
    main()