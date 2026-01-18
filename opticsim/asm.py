import torch


def fft2c(x: torch.Tensor) -> torch.Tensor:
    """Centered 2D FFT."""
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x)))


def ifft2c(X: torch.Tensor) -> torch.Tensor:
    """Centered 2D IFFT."""
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(X)))


def asm_transfer_function(
    N: int,
    dx: float,
    wavelength: float,
    z: float,
    device: torch.device,
    dtype: torch.dtype = torch.complex64,
    evanescent: str = "keep",  # "keep" or "cut"
) -> torch.Tensor:
    """
    Build ASM transfer function H(kx, ky) = exp(i kz z),
    kz = sqrt(k^2 - kx^2 - ky^2).

    evanescent:
      - "keep": keep complex kz (evanescent components decay naturally)
      - "cut":  remove evanescent components by masking (band-limit)
    """
    k = 2 * torch.pi / torch.tensor(wavelength, device=device, dtype=torch.float32)

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

    # keep: allow complex sqrt
    kz = torch.sqrt(kz2.to(dtype))
    H = torch.exp(1j * kz * z)
    return H.to(dtype)


def asm_propagate(U0: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """Propagate complex field U0 by applying H in frequency domain."""
    Uf = fft2c(U0)
    Uz = ifft2c(Uf * H)
    return Uz
