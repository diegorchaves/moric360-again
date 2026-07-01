"""
Combined WS-MSE + WS-SSIM loss for ERP (equirectangular projection) images.

All functions are implemented in pure PyTorch and are fully compatible with
autograd (i.e., they can be used directly inside a training loop).

Distortion term  : WS-MSE — spherically weighted mean squared error.
Perceptual term  : WS-SSIM loss — (1 − weighted SSIM).
Combined         : λ_dist · WS-MSE  +  λ_percep · (1 − WS-SSIM)

Default weights follow the original TensorFlow reference:
    C          = 0.1 × 2^{−5}  ≈ 3.125 × 10^{−3}
    C_D        = 0.75
    λ_dist     = C_D × C       ≈ 2.344 × 10^{−3}
    λ_percep   = 1.0

Notes
-----
- The WS-MSE normalisation is identical to ``compute_ws_mse`` in
  ``utils/eval_model.py``:  sum(w · sq_err) / (4π · 3)
- Expected input shape for all public functions: (H, W, 3), values in [0, 1].
- The SSIM branch uses a depthwise Gaussian conv (filter_size=11, σ=1.5).
  Images must have H ≥ 11 and W ≥ 11 — always true for ERP data.
"""

import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Default hyper-parameters (from the TF reference)
# ──────────────────────────────────────────────────────────────────────────────

_C_REF = 0.1 * (2.0 ** -5)          # ≈ 3.125e-3
_CD_REF = 0.75

LAMBDA_DIST_DEFAULT = _CD_REF * _C_REF   # ≈ 2.344e-3
LAMBDA_PERCEP_DEFAULT = 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Internal helper: spherical weights
# ──────────────────────────────────────────────────────────────────────────────

def _spherical_weights(
    H: int,
    W: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Per-row ERP spherical weights.

    For a full equirectangular image of size H × W:
        phi_i      = i · π / H          (i = 0 … H)
        w_row_i    = (2π / W) · (cos φ_i − cos φ_{i+1})

    Such that Σ_i w_row_i · W = 4π  (solid angle of the sphere).

    Returns
    -------
    Tensor of shape (H, 1, 1) — broadcasts over (W, C) automatically.
    """
    phis = torch.arange(H + 1, dtype=dtype, device=device) * torch.pi / H
    delta_theta = 2.0 * torch.pi / W
    column = delta_theta * (torch.cos(phis[:-1]) - torch.cos(phis[1:]))  # (H,)
    return column.view(H, 1, 1)


# ──────────────────────────────────────────────────────────────────────────────
# WS-MSE
# ──────────────────────────────────────────────────────────────────────────────

def compute_ws_mse_loss(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """
    Spherically weighted MSE for a full ERP image.

    Normalisation: sum(w · sq_err) / (4π · 3) — identical to the existing
    ``compute_ws_mse`` in ``utils/eval_model.py``.

    Parameters
    ----------
    img1, img2 : torch.Tensor
        Shape (H, W, 3), values in [0, 1], on the same device.

    Returns
    -------
    Scalar tensor.
    """
    H, W, _ = img1.shape
    w = _spherical_weights(H, W, img1.device, img1.dtype)  # (H, 1, 1)
    sq_err = (img1 - img2) ** 2                            # (H, W, 3)
    # For a full ERP image: W · Σ_h(w_h) = 4π  →  denominator = 4π · C
    return (sq_err * w).sum() / (4.0 * torch.pi * 3.0)


# ──────────────────────────────────────────────────────────────────────────────
# WS-SSIM loss
# ──────────────────────────────────────────────────────────────────────────────

def compute_ws_ssim_loss(
    img1: torch.Tensor,
    img2: torch.Tensor,
    filter_size: int = 11,
    filter_sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> torch.Tensor:
    """
    Differentiable WS-SSIM loss = 1 − WS-SSIM.

    Mirrors the manual SSIM map in the TF reference, then weights each pixel
    with the spherical weight of its row in the *full* image.

    Parameters
    ----------
    img1, img2    : torch.Tensor
        Shape (H, W, 3), values in [0, 1], on the same device.
    filter_size   : int
        Gaussian kernel size (default 11, standard SSIM).
    filter_sigma  : float
        Gaussian kernel σ (default 1.5).
    k1, k2        : float
        SSIM stability constants.

    Returns
    -------
    Scalar tensor in [0, 2].  Perfect reconstruction → 0.
    """
    H, W, C = img1.shape
    device = img1.device
    dtype = img1.dtype
    pad = filter_size // 2

    # ── reshape to (1, C, H, W) for conv2d ───────────────────────────────────
    i1 = img1.permute(2, 0, 1).unsqueeze(0)   # (1, C, H, W)
    i2 = img2.permute(2, 0, 1).unsqueeze(0)

    # ── depthwise Gaussian kernel ─────────────────────────────────────────────
    coords = torch.arange(filter_size, dtype=dtype, device=device) - pad
    gauss = torch.exp(-(coords ** 2) / (2.0 * filter_sigma ** 2))
    gauss = gauss / gauss.sum()
    k2d = torch.outer(gauss, gauss)                                          # (k, k)
    kernel = (
        k2d.view(1, 1, filter_size, filter_size)
        .expand(C, 1, filter_size, filter_size)
        .contiguous()
    )  # (C, 1, k, k)

    def _conv(x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, kernel, groups=C, padding=0)

    c1 = (k1 * 1.0) ** 2
    c2 = (k2 * 1.0) ** 2

    # ── SSIM statistics ───────────────────────────────────────────────────────
    mu1 = _conv(i1)            # (1, C, H_v, W_v)  where H_v = H − 2·pad
    mu2 = _conv(i2)
    mu1_sq  = mu1 * mu1
    mu2_sq  = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = _conv(i1 * i1) - mu1_sq
    sigma2_sq = _conv(i2 * i2) - mu2_sq
    sigma12   = _conv(i1 * i2) - mu1_mu2

    numerator   = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12   + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = numerator / denominator                    # (1, C, H_v, W_v)

    # ── spherical weights for the valid (SSIM) region ────────────────────────
    H_v = H - 2 * pad
    W_v = W - 2 * pad

    w_full  = _spherical_weights(H, W, device, dtype)    # (H,  1, 1)
    w_valid = w_full[pad: H - pad, :, :]                 # (H_v, 1, 1)

    # broadcast weight over (1, C, H_v, W_v)
    w4d = w_valid.view(1, 1, H_v, 1)

    weighted_ssim = ssim_map * w4d                        # (1, C, H_v, W_v)

    # Normalisation matches TF reference:
    #   reduce_sum(weighted_ssim) / (reduce_sum(weights_valid) × 3)
    # where reduce_sum(weights_valid) = Σ_h(w_h) × W_v
    ws_ssim = weighted_ssim.sum() / (w_valid.sum() * W_v * C)

    return 1.0 - ws_ssim


# ──────────────────────────────────────────────────────────────────────────────
# Combined loss (public API)
# ──────────────────────────────────────────────────────────────────────────────

def compute_combined_loss(
    img1: torch.Tensor,
    img2: torch.Tensor,
    lambda_dist: float = LAMBDA_DIST_DEFAULT,
    lambda_percep: float = LAMBDA_PERCEP_DEFAULT,
):
    """
    Combined WS-MSE + WS-SSIM loss for ERP images.

    Parameters
    ----------
    img1, img2    : torch.Tensor
        Shape (H, W, 3), values in [0, 1], on the same device.
    lambda_dist   : float
        Weight for the distortion (WS-MSE) term.
        Default ≈ 2.344e-3  (= 0.75 × 0.1 × 2^{-5}).
    lambda_percep : float
        Weight for the perceptual (WS-SSIM) term.
        Default = 1.0.

    Returns
    -------
    tuple of three scalar tensors:
        (total_loss, weighted_dist_loss, weighted_percep_loss)
    """
    dist_loss   = compute_ws_mse_loss(img1, img2)
    percep_loss = compute_ws_ssim_loss(img1, img2)

    weighted_dist   = lambda_dist   * dist_loss
    weighted_percep = lambda_percep * percep_loss

    return weighted_dist + weighted_percep, weighted_dist, weighted_percep
