"""Molecular encoders for REPA alignment."""

import torch
import torch.nn as nn
from typing import Optional

from tabasco.chem.convert import MoleculeConverter


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

    This encoder uses a simple MLP to encode coordinates. It is meant for
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


class ChemPropEncoder(MolecularEncoder):
    """Encoder using ChemProp's message passing neural network.

    ChemProp is a 2D graph-based encoder that operates on molecular topology.
    It does NOT use 3D coordinates - only atom types and bond connectivity.

    This encoder converts Tabasco's (coords, atomics) format to RDKit molecules,
    then uses ChemProp's BondMessagePassing to get atom-level embeddings.

    Note: Since ChemProp is graph-based (not 3D), the coords are only used
    to infer bond connectivity via RDKit's DetermineBonds algorithm.

    For REPA to work effectively, you should use pretrained weights (e.g., CheMeleon)
    so the encoder provides meaningful molecular representations as alignment targets.
    """

    def __init__(
        self,
        encoder_dim: int = 300,
        depth: int = 3,
        dropout: float = 0.0,
        pretrained: str = "chemeleon",
    ):
        """Initialize ChemProp encoder.

        Args:
            encoder_dim: Output dimension of message passing (ignored if using pretrained)
            depth: Number of message passing iterations (ignored if using pretrained)
            dropout: Dropout probability (ignored if using pretrained)
            pretrained: One of:
                - "chemeleon": Load CheMeleon foundation model (recommended, auto-downloads)
                - "none" or None: Random initialization (not recommended for REPA)
                - Path to a .pt file: Load custom pretrained weights
        """
        super().__init__()

        from chemprop.nn import BondMessagePassing
        from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer

        self.featurizer = SimpleMoleculeMolGraphFeaturizer()

        if pretrained is None or pretrained == "none":
            # Random initialization (not recommended for REPA)
            import warnings
            warnings.warn(
                "ChemPropEncoder initialized with random weights. "
                "For REPA to be effective, use pretrained='chemeleon' to load "
                "the CheMeleon foundation model with meaningful molecular representations."
            )
            self.encoder_dim = encoder_dim
            self.message_passing = BondMessagePassing(
                d_h=encoder_dim,
                depth=depth,
                dropout=dropout,
            )
        elif pretrained == "chemeleon":
            # Load CheMeleon foundation model
            chemeleon_weights = self._load_chemeleon()
            self.message_passing = BondMessagePassing(**chemeleon_weights["hyper_parameters"])
            self.message_passing.load_state_dict(chemeleon_weights["state_dict"])
            self.encoder_dim = chemeleon_weights["hyper_parameters"]["d_h"]
        else:
            # Load from custom path
            weights = torch.load(pretrained, map_location="cpu", weights_only=True)
            if "hyper_parameters" in weights:
                # ChemProp format
                self.message_passing = BondMessagePassing(**weights["hyper_parameters"])
                self.message_passing.load_state_dict(weights["state_dict"])
                self.encoder_dim = weights["hyper_parameters"]["d_h"]
            else:
                # Raw state dict
                self.encoder_dim = encoder_dim
                self.message_passing = BondMessagePassing(
                    d_h=encoder_dim,
                    depth=depth,
                    dropout=dropout,
                )
                self.message_passing.load_state_dict(weights)

        # Molecule converter for coords/atomics -> RDKit mol
        self.converter = MoleculeConverter()

    def _load_chemeleon(self) -> dict:
        """Download and load CheMeleon foundation model weights.

        CheMeleon is pretrained on 1M molecules from PubChem and provides
        meaningful molecular representations for REPA alignment.

        Returns:
            dict with "hyper_parameters" and "state_dict" keys
        """
        from pathlib import Path
        from urllib.request import urlretrieve
        import logging

        logger = logging.getLogger(__name__)

        ckpt_dir = Path.home() / ".chemprop"
        ckpt_dir.mkdir(exist_ok=True)
        model_path = ckpt_dir / "chemeleon_mp.pt"

        if not model_path.exists():
            logger.info(
                f"Downloading CheMeleon foundation model from Zenodo to {model_path}"
            )
            urlretrieve(
                "https://zenodo.org/records/15460715/files/chemeleon_mp.pt",
                model_path
            )
        else:
            logger.info(f"Loading cached CheMeleon from {model_path}")

        logger.info(
            "Please cite DOI: 10.48550/arXiv.2506.15792 when using CheMeleon in published work"
        )

        return torch.load(model_path, map_location="cpu", weights_only=True)

    def forward(self, coords, atomics, padding_mask):
        """Extract ChemProp atom embeddings.

        Args:
            coords: [B, N, 3] - atomic coordinates (used for bond inference)
            atomics: [B, N, atom_dim] - one-hot atom types
            padding_mask: [B, N] - True for padding positions

        Returns:
            repr: [B, N, encoder_dim] - atom-level representations
        """
        from tensordict import TensorDict
        from chemprop.data import BatchMolGraph

        B, N, _ = coords.shape
        device = coords.device

        # Convert each molecule in batch to RDKit mol, then to ChemProp MolGraph
        molgraphs = []
        atom_counts = []  # Track atoms per molecule for later padding

        for i in range(B):
            # Create TensorDict for this molecule
            mol_td = TensorDict({
                "coords": coords[i],
                "atomics": atomics[i],
                "padding_mask": padding_mask[i],
            })

            # Convert to RDKit mol (this handles bond inference)
            try:
                mol = self.converter.from_tensor(
                    mol_td,
                    rescale_coords=True,
                    sanitize=False,  # Don't sanitize to avoid failures
                    use_openbabel=False,  # Use simpler bond inference
                )
                if mol is not None:
                    mg = self.featurizer(mol)
                    molgraphs.append(mg)
                    atom_counts.append(mol.GetNumAtoms())
                else:
                    # Failed conversion - use placeholder
                    molgraphs.append(None)
                    atom_counts.append(0)
            except Exception:
                molgraphs.append(None)
                atom_counts.append(0)

        # Filter out failed conversions
        valid_mgs = [mg for mg in molgraphs if mg is not None]

        if len(valid_mgs) == 0:
            # All molecules failed - return zeros
            return torch.zeros(B, N, self.encoder_dim, device=device)

        # Create batch and run message passing
        bmg = BatchMolGraph(valid_mgs)
        bmg.to(device)

        with torch.no_grad():
            atom_embeddings = self.message_passing(bmg)  # [total_atoms, encoder_dim]

        # Reconstruct batched output with padding
        output = torch.zeros(B, N, self.encoder_dim, device=device)

        # Map embeddings back to batch positions
        valid_idx = 0
        emb_offset = 0
        for i in range(B):
            if molgraphs[i] is not None:
                n_atoms = atom_counts[valid_idx]
                # Copy atom embeddings to output, respecting padding
                output[i, :n_atoms] = atom_embeddings[emb_offset:emb_offset + n_atoms]
                emb_offset += n_atoms
                valid_idx += 1

        return output


class MACEEncoder(MolecularEncoder):
    """Wrapper around MACE encoder.

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
    embeddings, enabling alignment via cosine similarity or MSE loss.
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
