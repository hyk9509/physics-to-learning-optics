import torch
from torch import nn


def fft2c(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x)))


def ifft2c(X: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(X)))


def make_transfer_function(
    N: int,
    dx: float,
    wavelength: float,
    z: float,
    device: torch.device,
    dtype: torch.dtype = torch.complex64,
    evanescent: str = "keep",  # "keep" or "cut"
) -> torch.Tensor:
    """
    H = exp(i kz z), kz = sqrt(k^2 - kx^2 - ky^2)
    """
    # use float32 constants for stability
    wl = torch.tensor(wavelength, device=device, dtype=torch.float32)
    k = 2 * torch.pi / wl

    fx = torch.fft.fftshift(torch.fft.fftfreq(N, d=dx)).to(device)
    FX, FY = torch.meshgrid(fx, fx, indexing="xy")
    kx = 2 * torch.pi * FX
    ky = 2 * torch.pi * FY

    kz2 = (k * k) - (kx * kx + ky * ky)

    if evanescent == "cut":
        mask = kz2 >= 0
        kz = torch.zeros_like(kz2, dtype=dtype)
        kz = torch.where(mask, torch.sqrt(kz2.to(dtype)), kz)
        H = torch.exp(1j * kz * z) * mask
        return H.to(dtype)

    kz = torch.sqrt(kz2.to(dtype))
    H = torch.exp(1j * kz * z)
    return H.to(dtype)


class ASMPropagator(nn.Module):
    """
    미분 가능한 ASM forward model.
    - H는 buffer로 저장해서 매 forward마다 재계산하지 않음.
    - 입력 U0는 complex tensor.
    """

    def __init__(
        self,
        N: int,
        dx: float,
        wavelength: float,
        z: float,
        device: torch.device,
        dtype: torch.dtype = torch.complex64,
        evanescent: str = "keep",
    ):
        super().__init__()
        H = make_transfer_function(
            N=N, dx=dx, wavelength=wavelength, z=z,
            device=device, dtype=dtype, evanescent=evanescent
        )
        self.register_buffer("H", H)

    def forward(self, U0: torch.Tensor) -> torch.Tensor:
        Uf = fft2c(U0)
        Uz = ifft2c(Uf * self.H)
        return Uz
