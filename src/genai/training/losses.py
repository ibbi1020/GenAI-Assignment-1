"""Pixel and structural loss used by the restoration tasks."""

from __future__ import annotations

import torch
import torch.nn.functional as F

SSIM_WINDOW = 11
SSIM_SIGMA = 1.5
SSIM_C1 = 0.01**2
SSIM_C2 = 0.03**2


def reconstruction_loss(
    restored: torch.Tensor, clean: torch.Tensor, alpha: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return combined loss, mean L1, and mean SSIM.

    The combined loss is ``alpha * L1 + (1 - alpha) * (1 - SSIM)``.
    """
    if not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    l1 = (restored - clean).abs().mean()
    ssim = structural_similarity(restored, clean).mean()
    total = alpha * l1 + (1.0 - alpha) * (1.0 - ssim)
    return total, l1, ssim


def per_image_scores(restored: torch.Tensor, clean: torch.Tensor) -> dict[str, torch.Tensor]:
    """L1, SSIM, and PSNR for each image in a batch."""
    l1 = (restored - clean).abs().mean(dim=(1, 2, 3))
    ssim = structural_similarity(restored, clean)
    mse = (restored - clean).pow(2).mean(dim=(1, 2, 3)).clamp_min(1e-8)
    psnr = 10.0 * torch.log10(1.0 / mse)
    return {"l1": l1, "ssim": ssim, "psnr": psnr}


def validation_objective(mean_l1: float, mean_ssim: float) -> float:
    """Single score for Optuna. Lower means a better restoration."""
    return mean_l1 + (1.0 - mean_ssim)


def structural_similarity(restored: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """One SSIM value per image, averaged over channels and pixels."""
    if restored.shape != clean.shape:
        raise ValueError(f"shape mismatch: {tuple(restored.shape)} vs {tuple(clean.shape)}")
    if restored.ndim != 4:
        raise ValueError("expected NCHW tensors")

    channels = restored.size(1)
    window = _gaussian_window(SSIM_WINDOW, SSIM_SIGMA, channels, restored.device, restored.dtype)
    padding = SSIM_WINDOW // 2
    mu_restored = F.conv2d(restored, window, padding=padding, groups=channels)
    mu_clean = F.conv2d(clean, window, padding=padding, groups=channels)
    mu_restored_sq = mu_restored.pow(2)
    mu_clean_sq = mu_clean.pow(2)
    mu_cross = mu_restored * mu_clean

    sigma_restored_sq = (
        F.conv2d(restored * restored, window, padding=padding, groups=channels) - mu_restored_sq
    ).clamp_min(0)
    sigma_clean_sq = (
        F.conv2d(clean * clean, window, padding=padding, groups=channels) - mu_clean_sq
    ).clamp_min(0)
    sigma_cross = F.conv2d(restored * clean, window, padding=padding, groups=channels) - mu_cross

    numerator = (2 * mu_cross + SSIM_C1) * (2 * sigma_cross + SSIM_C2)
    denominator = (mu_restored_sq + mu_clean_sq + SSIM_C1) * (sigma_restored_sq + sigma_clean_sq + SSIM_C2)
    return (numerator / denominator).mean(dim=(1, 2, 3))


def _gaussian_window(
    window_size: int, sigma: float, channels: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    kernel = torch.exp(-(coords**2) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    window_2d = torch.outer(kernel, kernel)
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()
