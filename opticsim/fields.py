import torch


def make_grid(N: int, dx: float, device: torch.device):
    x = (torch.arange(N, device=device) - N // 2) * dx
    X, Y = torch.meshgrid(x, x, indexing="xy")
    return X, Y


def plane_wave(N: int, device: torch.device, dtype: torch.dtype):
    return torch.ones((N, N), device=device, dtype=dtype)


def single_slit_aperture(X: torch.Tensor, slit_width: float) -> torch.Tensor:
    # aperture is real-valued mask (0 or 1)
    return (torch.abs(X) <= (slit_width / 2)).to(dtype=torch.float32)


def apply_aperture(U: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    # A will be promoted to complex dtype automatically
    return U * A
