"""
contrastive_model.py
--------------------
SimCLR-inspired model: a ResNet18 encoder with a 2-layer MLP projection head,
plus the NT-Xent (Normalized Temperature-scaled Cross Entropy) contrastive loss.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Projection Head
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """Two-layer MLP projection head that maps encoder features -> embedding space.

    SimCLR trains via the projected representations but the encoder's output
    (before the projection) is used for downstream tasks.

    Args:
        input_dim:  Dimensionality of the encoder's output features.
        hidden_dim: Hidden layer size.
        output_dim: Final embedding dimensionality (128 in the SimCLR paper).
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 512, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# SimCLR Model
# ---------------------------------------------------------------------------

class SimCLRModel(nn.Module):
    """ResNet18 encoder + MLP projection head for contrastive pre-training.

    Args:
        embedding_dim:  Output dimension of the projection head.
        pretrained:     Load ImageNet weights for the ResNet18 backbone.
        freeze_backbone: Freeze backbone weights and only train the head.
                         Useful for fine-tuning on small datasets.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────────────────────
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        encoder_out_dim = backbone.fc.in_features          # 512 for ResNet18

        # Remove the original classification head
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])

        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info("Backbone frozen — only projection head will be trained.")

        # ── Projection Head ───────────────────────────────────────────────────
        self.projector = ProjectionHead(
            input_dim=encoder_out_dim,
            hidden_dim=encoder_out_dim,
            output_dim=embedding_dim,
        )

        logger.info(
            "SimCLRModel built: ResNet18 encoder (pretrained=%s) -> ProjectionHead -> %d-d embedding.",
            pretrained, embedding_dim,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both encoder features and projected embeddings.

        Args:
            x: Input image tensor of shape (B, 3, H, W).

        Returns:
            Tuple of:
                features   (B, 512)  — used for downstream classification
                embeddings (B, embedding_dim) — used for contrastive loss
        """
        features = self.encoder(x)                 # (B, 512, 1, 1)
        features = features.squeeze(-1).squeeze(-1)  # (B, 512)
        embeddings = self.projector(features)        # (B, 128)
        return features, embeddings

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience method: return only encoder features (no projection).

        Used during embedding extraction and downstream classification.
        """
        with torch.no_grad():
            features = self.encoder(x)
            return features.squeeze(-1).squeeze(-1)


# ---------------------------------------------------------------------------
# NT-Xent Contrastive Loss
# ---------------------------------------------------------------------------

class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss (SimCLR's loss).

    Given a batch of 2N augmented views (N pairs), the objective is to pull
    corresponding views together and push all other views apart.

    Args:
        temperature: Softmax temperature τ (lower -> sharper distribution).
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """Compute NT-Xent loss for a batch of view pairs.

        Args:
            z_i: Embeddings for view 1, shape (N, D).
            z_j: Embeddings for view 2, shape (N, D).

        Returns:
            Scalar loss tensor.
        """
        N = z_i.size(0)
        device = z_i.device

        # ℓ2-normalise both sets of embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Concatenate: [z_i; z_j] -> (2N, D)
        z = torch.cat([z_i, z_j], dim=0)

        # Pairwise cosine similarity matrix (2N, 2N)
        sim = torch.mm(z, z.T) / self.temperature

        # Mask out diagonal (self-similarity)
        mask = torch.eye(2 * N, dtype=torch.bool, device=device)
        sim = sim.masked_fill(mask, float("-inf"))

        # Positive pair indices
        # For z_i[k], its positive is z_j[k] located at index N+k in z
        labels = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(device)

        loss = F.cross_entropy(sim, labels)
        return loss


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    model = SimCLRModel(embedding_dim=128, pretrained=False).to(device)
    loss_fn = NTXentLoss(temperature=0.5)

    # Dummy batch: 8 images, 3 channels, 224×224
    B = 8
    x = torch.randn(B, 3, 224, 224).to(device)

    feat, emb = model(x)
    logger.info("Encoder features: %s", feat.shape)
    logger.info("Projected embeddings: %s", emb.shape)

    # Simulate two views
    _, z_i = model(x)
    _, z_j = model(x)
    loss = loss_fn(z_i, z_j)
    logger.info("NT-Xent loss: %.4f", loss.item())
    logger.info("contrastive_model.py smoke-test passed.")
