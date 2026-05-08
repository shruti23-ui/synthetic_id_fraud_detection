"""
two_stream_model.py
-------------------
Two-stream RGB + frequency-domain forgery detector with per-branch
transformer fusion. Designed specifically for ID-document forgery
where artefacts often live in the frequency domain (JPEG re-compression,
inpainting noise, copy-paste discontinuities).

Architecture:
  Input (B, 3, 224, 224)  -- RGB, ImageNet-normalised
       |
       +--------------------- RGB BRANCH ---------------------+
       |                                                       |
       |   ResNet50 (ImageNet V2)  ->  (B, 2048, 7, 7)        |
       |   1x1 conv -> 256                                     |
       |   reshape to (B, 49, 256)                             |
       |   + learnable CLS token + pos enc -> (B, 50, 256)     |
       |   2-layer TransformerEncoder (4 heads, GELU)          |
       |   take CLS -> (B, 256)                                |
       |                                                       |
       +--------------------- FFT BRANCH ---------------------+
       |                                                       |
       |   x_fft = log(1 + |FFT(x)|), then fftshift           |
       |   ResNet18 (ImageNet)     ->  (B, 512, 7, 7)         |
       |   1x1 conv -> 256                                     |
       |   reshape to (B, 49, 256)                             |
       |   + learnable CLS token + pos enc -> (B, 50, 256)     |
       |   2-layer TransformerEncoder (4 heads, GELU)          |
       |   take CLS -> (B, 256)                                |
       |                                                       |
       +-------------------- FUSION HEAD ---------------------+
       |                                                       |
       |   concat (B, 512)                                     |
       |   LayerNorm + Dropout                                 |
       |   Linear 512 -> 256 + GELU + Dropout                  |
       |   Linear 256 -> 2 logits                              |
       +-------------------------------------------------------+

Key design choices
------------------
* Two branches use *different* backbones (R50 for RGB, R18 for FFT).
  The frequency-domain image is simpler (log-magnitude is essentially
  grayscale + symmetric structure) so a smaller backbone suffices.

* Per-branch transformer over the 7x7 spatial grid lets each branch
  do spatial reasoning ("which patch is forged?") instead of just
  global pooling.

* Late fusion at the CLS-token level. We deliberately do NOT cross-
  attend between branches mid-stream because the streams have very
  different feature distributions and that often hurts at small data.

* GELU + LayerNorm + pre-norm transformer (standard modern recipe).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
)


def _strip_classifier(backbone: nn.Module) -> nn.Sequential:
    """Drop avgpool + fc from a torchvision ResNet so we keep the
    spatial (B, C, H, W) feature map."""
    return nn.Sequential(*list(backbone.children())[:-2])


def _compute_fft_magnitude(x: torch.Tensor) -> torch.Tensor:
    """log(1 + |FFT(x)|) per channel, with the DC component centred.

    Input  (B, C, H, W) - ImageNet-normalised RGB
    Output (B, C, H, W) - real-valued log-magnitude spectrogram.

    Notes:
      * We FFT the *normalised* image. The absolute scale doesn't matter
        because the FFT is linear and we log-compress immediately afterwards.
      * fftshift puts the DC component (low freq) at the centre of the
        feature map -- helps the convolutional inductive bias.
      * Computed in fp32 even under AMP because torch.fft prefers float32.
    """
    with torch.amp.autocast(device_type="cuda", enabled=False):
        x_f = x.float()
        spec = torch.fft.fft2(x_f, dim=(-2, -1))
        mag  = torch.abs(spec)
        log_mag = torch.log1p(mag)
        log_mag = torch.fft.fftshift(log_mag, dim=(-2, -1))
    return log_mag


class TransformerBranch(nn.Module):
    """Take a (B, C, H, W) feature map, reshape to a sequence of HW patch
    tokens, prepend a learnable CLS token + positional encoding, run a
    small Transformer encoder, return the CLS token."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        spatial: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        n_tokens = spatial * spatial + 1
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, embed_dim))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, fmap: torch.Tensor) -> torch.Tensor:
        # fmap: (B, C, H, W)
        x = self.proj(fmap)                       # (B, embed_dim, H, W)
        B, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)          # (B, H*W, embed_dim)
        cls = self.cls.expand(B, -1, -1)          # (B, 1, embed_dim)
        x = torch.cat([cls, x], dim=1) + self.pos # (B, H*W+1, embed_dim)
        x = self.encoder(x)
        x = self.norm(x)
        return x[:, 0]                            # CLS token: (B, embed_dim)


class TwoStreamForgeryNet(nn.Module):
    """RGB + FFT two-stream model with per-branch transformer fusion."""

    def __init__(
        self,
        num_classes: int = 2,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_transformer_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        # ── RGB BRANCH ─────────────────────────────────────────────────
        rgb_back = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.rgb_features = _strip_classifier(rgb_back)         # (B, 2048, 7, 7)
        self.rgb_branch = TransformerBranch(
            in_channels=2048, embed_dim=embed_dim,
            num_heads=num_heads, num_layers=num_transformer_layers,
            spatial=7, dropout=dropout * 0.5,
        )

        # ── FFT BRANCH ─────────────────────────────────────────────────
        # ResNet18 because the frequency-domain image is simpler.
        # ImageNet weights are still a useful init even though the input
        # distribution is different — bn1 will adapt during fine-tuning.
        fft_back = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.fft_features = _strip_classifier(fft_back)          # (B, 512, 7, 7)
        # Normalise the FFT input distribution before the backbone sees it.
        # An input BN learns per-channel mean/std for log-magnitude FFT.
        self.fft_input_norm = nn.BatchNorm2d(3)
        self.fft_branch = TransformerBranch(
            in_channels=512, embed_dim=embed_dim,
            num_heads=num_heads, num_layers=num_transformer_layers,
            spatial=7, dropout=dropout * 0.5,
        )

        # ── FUSION HEAD ────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(2 * embed_dim),
            nn.Dropout(dropout),
            nn.Linear(2 * embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 224, 224) -- ImageNet-normalised RGB

        # ── RGB stream ────────────────────────────────────────────────
        rgb_fmap = self.rgb_features(x)            # (B, 2048, 7, 7)
        rgb_cls  = self.rgb_branch(rgb_fmap)       # (B, embed_dim)

        # ── FFT stream ────────────────────────────────────────────────
        x_fft = _compute_fft_magnitude(x)          # (B, 3, 224, 224)
        x_fft = self.fft_input_norm(x_fft)
        fft_fmap = self.fft_features(x_fft)        # (B, 512, 7, 7)
        fft_cls  = self.fft_branch(fft_fmap)       # (B, embed_dim)

        # ── Fusion ────────────────────────────────────────────────────
        fused = torch.cat([rgb_cls, fft_cls], dim=1)   # (B, 2*embed_dim)
        return self.classifier(fused)


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    log = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoStreamForgeryNet(num_classes=2).to(device)
    n = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("TwoStreamForgeryNet: %s total / %s trainable params",
             f"{n:,}", f"{n_train:,}")

    x = torch.randn(2, 3, 224, 224, device=device)
    y = model(x)
    log.info("forward OK: input=%s -> output=%s", tuple(x.shape), tuple(y.shape))

    # quick gradient check
    target = torch.randint(0, 2, (2,), device=device)
    loss = nn.functional.cross_entropy(y, target)
    loss.backward()
    log.info("backward OK: loss=%.4f", loss.item())
