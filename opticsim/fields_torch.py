import torch


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_grid(N: int, dx: float, device: torch.device):
    x = (torch.arange(N, device=device) - N // 2) * dx
    X, Y = torch.meshgrid(x, x, indexing="xy")
    return X, Y


def target_from_slit(N: int, dx: float, device: torch.device, slit_width: float) -> torch.Tensor:
    """
    간단한 타깃 강도(target intensity)를 만들기 위한 함수.
    여기서는 단일 슬릿 마스크를 그냥 타깃 이미지로 사용한다(학습/최적화 검증용).
    """
    X, _ = make_grid(N, dx, device)
    A = (torch.abs(X) <= slit_width / 2).to(torch.float32)
    return A  # (N,N) float32


def normalize01(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x = x - x.min()
    return x / (x.max() + eps)
