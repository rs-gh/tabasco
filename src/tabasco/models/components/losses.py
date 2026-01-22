from typing import Optional, Callable
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import Tensor, nn

from tabasco.flow.path import FlowPath
from tabasco.utils.metric_utils import split_losses_by_time


class InterDistancesLoss(nn.Module):
    """Mean-squared error between predicted and reference inter-atomic distance matrices."""

    def __init__(
        self,
        distance_threshold: Optional[float] = None,
        sqrd: bool = False,
        key: str = "coords",
        key_pad_mask: str = "padding_mask",
        time_factor: Optional[Callable] = None,
    ):
        """Initialize the loss module.

        Args:
            distance_threshold: If provided, only atom pairs with distance <= threshold
                contribute to the loss. Units must match the coordinate system.
            sqrd: When `True` the raw *squared* distances are used instead of their square-root.
                Set this to `True` if you have pre-squared your training targets.
            key: Key that stores coordinates inside `TensorDict` objects.
            key_pad_mask: Key that stores the boolean padding mask inside `TensorDict` objects.
            time_factor: Optional callable `f(t)` that rescales the per-pair loss as a
                function of the interpolation time `t`.
        """
        super().__init__()
        self.key = key
        self.key_pad_mask = key_pad_mask
        self.distance_threshold = distance_threshold
        self.sqrd = sqrd
        self.mse_loss = nn.MSELoss(reduction="none")
        self.time_factor = time_factor

    def inter_distances(self, coords1, coords2, eps: float = 1e-6) -> Tensor:
        """Compute pairwise distances between two coordinate sets.

        Args:
            coords1: Coordinate tensor of shape `(N, 3)`.
            coords2: Coordinate tensor of shape `(M, 3)`.
            eps: Numerical stability term added before `sqrt` when `sqrd` is `False`.

        Returns:
            Tensor of shape `(N, M)` containing pairwise distances. Values are squared
            distances when the instance was created with `sqrd=True`.
        """
        if self.sqrd:
            return torch.cdist(coords1, coords2, p=2) ** 2
        else:
            squared_dist = torch.cdist(coords1, coords2, p=2) ** 2
            return torch.sqrt(squared_dist + eps)

    def forward(
        self, path: FlowPath, pred: TensorDict, compute_stats: bool = True
    ) -> Tensor:
        """Compute the inter-distance MSE loss.

        Args:
            path: `FlowPath` containing ground-truth endpoint coordinates and the
                interpolation time `t`.
            pred: `TensorDict` with predicted coordinates under the same `key` as
                specified during initialization.
            compute_stats: If `True` additionally returns distance-loss statistics binned
                by time for logging purposes.

        Returns:
            - loss:         Scalar tensor with the mean loss.
            - stats_dict:   Dictionary with binned loss statistics. Empty when
                `compute_stats` is `False`.
        """
        real_mask = 1 - path.x_1[self.key_pad_mask].float()
        real_mask = real_mask.unsqueeze(-1)

        pred_coords = pred[self.key]
        true_coords = path.x_1[self.key]

        pred_dists = self.inter_distances(pred_coords, pred_coords)
        true_dists = self.inter_distances(true_coords, true_coords)

        mask_2d = torch.matmul(real_mask, real_mask.transpose(-1, -2))

        # Add distance threshold mask (0 for pairs where distance > threshold)
        if self.distance_threshold is not None:
            distance_mask = (true_dists <= self.distance_threshold).float()
            combined_mask = mask_2d * distance_mask
        else:
            combined_mask = mask_2d

        dists_loss = self.mse_loss(pred_dists, true_dists)
        dists_loss = dists_loss * combined_mask

        if self.time_factor:
            dists_loss = dists_loss * self.time_factor(path.t)

        if compute_stats:
            binned_losses = split_losses_by_time(path.t, dists_loss, 5)
            stats_dict = {
                **{f"dists_loss_bin_{i}": loss for i, loss in enumerate(binned_losses)},
            }
        else:
            stats_dict = {}

        dists_loss = dists_loss.mean()
        return dists_loss, stats_dict


class REPALoss(nn.Module):
    """Representation Alignment loss between diffusion hidden states and frozen encoder.

    Following the REPA paper (https://arxiv.org/pdf/2410.06940), this loss aligns
    the diffusion model's internal representations with a pre-trained molecular encoder.
    This provides semantic guidance that improves both generation quality and training
    efficiency.

    The key idea: Align the diffusion model's hidden states (from noisy molecules at time t)
    with the frozen encoder's embeddings (from clean molecules at t=1). This helps the
    diffusion model learn better semantic representations.
    """

    def __init__(
        self,
        encoder: nn.Module,
        projector: nn.Module,
        lambda_repa: float = 0.5,
        time_weighting: bool = False,
    ):
        """Initialize REPA loss.

        Args:
            encoder: Frozen pre-trained molecular encoder (e.g., MACE, ChemProp, DummyEncoder)
            projector: Trainable projection layer (hidden_dim -> encoder_dim)
            lambda_repa: Weight for REPA loss relative to flow matching loss
            time_weighting: If True, apply higher weight to alignment at t~1 (clean molecules)
        """
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.lambda_repa = lambda_repa
        self.time_weighting = time_weighting

        # Freeze encoder to prevent co-adaptation
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(
        self,
        path: FlowPath,
        pred: TensorDict,
        compute_stats: bool = True
    ):
        """Compute REPA alignment loss.

        Args:
            path: FlowPath containing x_1 (clean molecules) and t (time)
            pred: Predictions containing hidden_states from diffusion model
            compute_stats: Whether to compute detailed statistics

        Returns:
            (loss, stats_dict)
        """
        # If hidden states not available, return zero loss
        if "hidden_states" not in pred:
            return 0.0, {}

        hidden_states = pred["hidden_states"]  # [B, N, hidden_dim]
        padding_mask = pred["padding_mask"]    # [B, N]

        # Get target representations from frozen encoder
        # IMPORTANT: Use CLEAN molecules (x_1) as target, NOT noisy x_t
        # Encoder is trained on clean data and won't give meaningful embeddings for noise
        with torch.no_grad():
            target_repr = self.encoder(
                path.x_1["coords"],  # Clean coordinates
                path.x_1["atomics"], # Clean atom types
                padding_mask
            )  # [B, N, encoder_dim]

        # Project diffusion hidden states to encoder space
        projected_repr = self.projector(hidden_states)  # [B, N, encoder_dim]

        # Compute MSE loss, masking padded positions
        real_mask = ~padding_mask  # [B, N]
        loss = F.mse_loss(
            projected_repr[real_mask],
            target_repr[real_mask]
        )

        # Optional: weight by time (stronger alignment for cleaner molecules)
        if self.time_weighting:
            # t close to 1 → clean molecule → higher weight
            # t close to 0 → noise → lower weight
            # Simple approach: use mean time as weight
            time_weight = path.t.mean()
            loss = loss * time_weight

        # Scale by lambda
        loss = self.lambda_repa * loss

        stats_dict = {}
        if compute_stats:
            stats_dict["repa_loss"] = loss.item()

            # Compute cosine similarity as alignment metric
            # Higher values (closer to 1) indicate better alignment
            with torch.no_grad():
                proj_mean = projected_repr[real_mask].mean(dim=0)
                target_mean = target_repr[real_mask].mean(dim=0)
                alignment = F.cosine_similarity(
                    proj_mean.unsqueeze(0),
                    target_mean.unsqueeze(0),
                    dim=1
                ).item()
                stats_dict["repa_alignment"] = alignment

        return loss, stats_dict
