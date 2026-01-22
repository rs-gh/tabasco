"""Molecular encoders for REPA alignment."""

import torch.nn as nn


class MolecularEncoder(nn.Module):
    """Base class for frozen molecular encoders used in REPA."""

    def forward(self, coords, atomics, padding_mask):
        """Extract representations from molecules.

        Args:
            coords: [B, N, 3] - atomic coordinates
            atomics: [B, N, atom_dim] - one-hot atom types
            padding_mask: [B, N] - True for padding

        Returns:
            repr: [B, N, encoder_dim] - molecular representations
        """
        raise NotImplementedError


class DummyEncoder(MolecularEncoder):
    """Simple encoder for testing REPA integration without external dependencies.

    This encoder uses a simple MLP to encode coordinates. It's meant for
    testing the REPA integration pipeline before adding a real molecular
    encoder like MACE or ChemProp.
    """

    def __init__(
        self, input_dim: int = 3, hidden_dim: int = 128, encoder_dim: int = 256
    ):
        """Args:
        input_dim: Input dimension (default 3 for coordinates)
        hidden_dim: Hidden layer dimension
        encoder_dim: Output embedding dimension
        """
        super().__init__()
        self.encoder_dim = encoder_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, encoder_dim),
        )

    def forward(self, coords, atomics, padding_mask):
        """Encode coordinates with simple MLP.

        Args:
            coords: [B, N, 3] - atomic coordinates
            atomics: [B, N, atom_dim] - one-hot atom types (unused in this simple encoder)
            padding_mask: [B, N] - True for padding

        Returns:
            repr: [B, N, encoder_dim] - molecular representations
        """
        # Simple: just encode coordinates
        # In reality, encoder would use both coords and atom types
        repr = self.mlp(coords)  # [B, N, encoder_dim]

        # Mask padded positions
        real_mask = ~padding_mask
        repr = repr * real_mask.unsqueeze(-1)

        return repr


class MACEEncoder(MolecularEncoder):
    """Wrapper around MACE encoder (requires mace-torch package).

    MACE is a physics-based molecular encoder that uses equivariant message
    passing. It's well-suited for molecular property prediction.

    Note: This is a placeholder. Full implementation requires:
    1. Installing mace-torch package
    2. Loading pre-trained MACE weights
    3. Converting TABASCO's data format to MACE's expected format
    """

    def __init__(self, pretrained_path: str = None):
        """Args:
        pretrained_path: Path to pre-trained MACE checkpoint
        """
        super().__init__()
        # TODO: Import and initialize MACE
        # from mace import MACE
        # self.mace = MACE.load(pretrained_path)
        raise NotImplementedError("MACE integration not yet implemented")

    def forward(self, coords, atomics, padding_mask):
        """Extract MACE embeddings.

        Args:
            coords: [B, N, 3] - atomic coordinates
            atomics: [B, N, atom_dim] - one-hot atom types
            padding_mask: [B, N] - True for padding

        Returns:
            repr: [B, N, encoder_dim] - MACE representations
        """
        # TODO: Convert to MACE format and get embeddings
        raise NotImplementedError


class Projector(nn.Module):
    """Projects diffusion hidden states to encoder embedding space.

    This is a trainable projection layer that maps the diffusion model's
    hidden states to the same dimensionality as the frozen encoder's
    embeddings, enabling alignment via MSE loss.
    """

    def __init__(self, hidden_dim: int, encoder_dim: int, num_layers: int = 2):
        """Args:
        hidden_dim: Dimension of diffusion model hidden states
        encoder_dim: Dimension of encoder embeddings
        num_layers: Number of MLP layers
        """
        super().__init__()

        layers = []
        in_dim = hidden_dim
        for _ in range(num_layers - 1):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.SiLU(),
                ]
            )
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, encoder_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, hidden_states):
        """Project hidden states to encoder space.

        Args:
            hidden_states: [B, N, hidden_dim]

        Returns:
            projected: [B, N, encoder_dim]
        """
        return self.mlp(hidden_states)
