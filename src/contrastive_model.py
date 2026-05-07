"""
contrastive_model.py
--------------------
SimCLR-inspired model and the contrastive losses used here.

Modules:
  ProjectionHead       : 2-layer MLP from encoder features to embedding space
  ClassificationHead   : optional 2-layer MLP for the multi-task ctype head
  SimCLRModel          : ResNet18 encoder + projection head (+ optional ctype head)
  NTXentLoss           : standard SimCLR self-supervised contrastive loss
  SupConLoss           : supervised contrastive loss (Khosla et al., NeurIPS 2020)
                         pulls together all samples sharing a label, pushes apart
                         the rest. Uses fake/real OR ctype labels.
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
        # BatchNorm in the projection head is critical for SimCLR — it forces
        # cross-sample variance and prevents representation collapse (the
        # failure mode where the encoder maps every input to the same
        # constant vector and the contrastive loss sticks at log(2N-1)).
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

class ClassificationHead(nn.Module):
    """Small MLP head for the multi-task ctype prediction.

    Predicts forgery type (real / Inpaint_and_Rewrite / Crop_and_Replace).
    Used as an auxiliary task alongside contrastive learning to inject
    forgery-aware supervision into the encoder.
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimCLRModel(nn.Module):
    """ResNet18 encoder + MLP projection head (+ optional multi-task head).

    Args:
        embedding_dim:    Output dim of the projection head (used by NT-Xent / SupCon).
        pretrained:       Load ImageNet weights for the ResNet18 backbone.
        freeze_backbone:  Freeze backbone weights and only train heads.
        num_ctypes:       If > 0, attach a ClassificationHead that predicts
                          forgery-type logits during training.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        num_ctypes: int = 0,
    ):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────────────────────
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        encoder_out_dim = backbone.fc.in_features          # 512 for ResNet18
        self.encoder_out_dim = encoder_out_dim

        # Remove the original classification head
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])

        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info("Backbone frozen - only heads will be trained.")

        # ── Projection Head (contrastive) ────────────────────────────────────
        self.projector = ProjectionHead(
            input_dim=encoder_out_dim,
            hidden_dim=encoder_out_dim,
            output_dim=embedding_dim,
        )

        # ── Optional multi-task ctype head ───────────────────────────────────
        self.num_ctypes = num_ctypes
        self.ctype_head = (
            ClassificationHead(encoder_out_dim, encoder_out_dim // 2, num_ctypes)
            if num_ctypes > 0 else None
        )

        logger.info(
            "SimCLRModel built: ResNet18(pretrained=%s) -> ProjHead(%d) %s",
            pretrained, embedding_dim,
            f"+ CTypeHead({num_ctypes})" if num_ctypes > 0 else "",
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Standard forward returning (encoder_features, projected_embeddings)."""
        features = self.encoder(x)
        features = features.squeeze(-1).squeeze(-1)
        embeddings = self.projector(features)
        return features, embeddings

    def forward_multitask(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward returning (features, embeddings, ctype_logits).

        Raises if the model was built without a ctype head.
        """
        if self.ctype_head is None:
            raise RuntimeError("Model has no ctype head; rebuild with num_ctypes > 0.")
        features = self.encoder(x).squeeze(-1).squeeze(-1)
        embeddings = self.projector(features)
        ctype_logits = self.ctype_head(features)
        return features, embeddings, ctype_logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: return only encoder features (no projection, no grad)."""
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
# Supervised Contrastive Loss (Khosla et al., NeurIPS 2020)
# ---------------------------------------------------------------------------

class SupConLoss(nn.Module):
    """Supervised contrastive loss.

    Generalises NT-Xent: for each anchor, all samples in the batch sharing
    the same label are treated as positives (in addition to the augmentation
    pair). Uses fake/real OR ctype labels.

    Args:
        temperature: Softmax temperature.

    Shape:
        z_i, z_j: (N, D) embedding pairs.
        labels:   (N,)   integer class labels (same for both views).

    Reference:
        Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020.
        https://arxiv.org/abs/2004.11362
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        N = z_i.size(0)
        device = z_i.device

        # ℓ2-normalise
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # (2N, D) features and (2N,) labels
        features = torch.cat([z_i, z_j], dim=0)
        all_labels = torch.cat([labels, labels], dim=0).view(-1, 1)

        # Pairwise similarity (logits) (2N, 2N)
        logits = torch.matmul(features, features.T) / self.temperature

        # Stabilise softmax
        logits_max, _ = logits.max(dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # Mask: which (i, j) pairs share a label  (i != j)
        pos_mask = (all_labels == all_labels.T).float().to(device)
        # Self-mask: zero out the diagonal
        eye = torch.eye(2 * N, dtype=torch.float, device=device)
        pos_mask = pos_mask - eye

        # Denominator: all (i, k) pairs with k != i
        logits_mask = 1.0 - eye
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # Mean log-prob over positives for each anchor
        pos_count = pos_mask.sum(dim=1).clamp(min=1.0)
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / pos_count

        loss = -mean_log_prob_pos.mean()
        return loss


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    B = 8
    x = torch.randn(B, 3, 224, 224).to(device)

    # 1) Standard SimCLR (no ctype head)
    model = SimCLRModel(embedding_dim=128, pretrained=False, num_ctypes=0).to(device)
    feat, emb = model(x)
    logger.info("Standard: feat=%s emb=%s", feat.shape, emb.shape)

    nt = NTXentLoss(temperature=0.5)
    _, z1 = model(x); _, z2 = model(x)
    logger.info("NT-Xent loss: %.4f", nt(z1, z2).item())

    sc = SupConLoss(temperature=0.1)
    labels = torch.randint(0, 2, (B,), device=device)
    logger.info("SupCon loss : %.4f", sc(z1, z2, labels).item())

    # 2) Multi-task (with ctype head)
    mt = SimCLRModel(embedding_dim=128, pretrained=False, num_ctypes=3).to(device)
    feat, emb, ctype_logits = mt.forward_multitask(x)
    logger.info("Multi-task: feat=%s emb=%s ctype_logits=%s",
                feat.shape, emb.shape, ctype_logits.shape)
    logger.info("contrastive_model.py smoke-test passed.")
