"""Molecular encoders for REPA alignment."""

import torch
import torch.nn as nn

from tabasco.chem.convert import MoleculeConverter


class MolecularEncoder(nn.Module):
    """Base class for frozen molecular encoders used in REPA."""

    def forward(self, coords, atomics, padding_mask, smiles=None):
        """Extract representations from molecules.

        Args:
            coords: [B, N, 3] - atomic coordinates
            atomics: [B, N, atom_dim] - one-hot atom types
            padding_mask: [B, N] - True for padding
            smiles: Optional list of SMILES strings for fast-path encoders

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

    def forward(self, coords, atomics, padding_mask, smiles=None):
        """Encode coordinates with simple MLP.

        Args:
            coords: [B, N, 3] - atomic coordinates
            atomics: [B, N, atom_dim] - one-hot atom types (unused in this simple encoder)
            padding_mask: [B, N] - True for padding
            smiles: Ignored (accepted for interface compatibility with ChemPropEncoder)

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

    For REPA to work effectively, we use pretrained weights (e.g., CheMeleon)
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

        # coords/atomics -> RDKit.Chem.Mol
        self.converter = MoleculeConverter()
        # RDKit.Chem.Mol -> ChemProp MolGraph featurizer
        self.featurizer = SimpleMoleculeMolGraphFeaturizer()
        # SMILES -> MolGraph cache (exact: ChemProp is 2-D, same graph for all conformers)
        self._molgraph_cache = {}

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
            # Following the example at:
            # https://github.com/chemprop/chemprop/blob/main/examples/chemeleon_foundation_finetuning.ipynb
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

    def _load_chemeleon(self) -> dict:
        """Download and load CheMeleon foundation model weights.
        Paper: https://arxiv.org/pdf/2506.15792

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

        return torch.load(model_path, map_location="cpu", weights_only=True)

    def forward(self, coords, atomics, padding_mask, smiles=None):
        """Extract ChemProp atom embeddings.

        Args:
            coords: [B, N, 3] - atomic coordinates
            atomics: [B, N, atom_dim] - one-hot atom types
            padding_mask: [B, N] - True for padding positions
            smiles: optional list of B canonical SMILES strings. When provided,
                molecules are built via Chem.MolFromSmiles (fast, ~0.4 ms each)
                and MolGraphs are cached, avoiding 3-D bond inference (~20 ms
                each) entirely. Falls back to from_tensor when None.

        Returns:
            repr: [B, N, encoder_dim] - atom-level representations
        """
        from tensordict import TensorDict
        from chemprop.data import BatchMolGraph
        from rdkit import Chem

        B, N, _ = coords.shape
        device = coords.device

        # RDKit/ChemProp operations are CPU-bound. Moving tensors to CPU once
        # here avoids ~25,000 implicit GPU→CPU synchronisations per forward
        # pass that would otherwise arise from .item() calls and element-wise
        # iteration over CUDA tensors inside _make_mol_simple_imputation and
        # _get_atom_types (each call to coords[i,j].item() stalls the GPU).
        coords = coords.cpu()
        atomics = atomics.cpu()
        padding_mask = padding_mask.cpu()

        # Convert each molecule in batch to RDKit mol, then to ChemProp MolGraph
        molgraphs = []
        atom_counts = []  # Track atoms per molecule for later padding

        for i in range(B):
            smi = smiles[i] if smiles is not None else None

            # Fast path: build from SMILES and use per-SMILES MolGraph cache.
            # ChemProp is a 2-D GNN so the MolGraph is identical for every
            # conformer of the same molecule — caching is exact, not approximate.
            if smi is not None:
                cached = self._molgraph_cache.get(smi)
                if cached is not None:
                    molgraphs.append(cached)
                    atom_counts.append(cached.V.shape[0])
                    continue
                try:
                    mol = Chem.MolFromSmiles(smi)
                    if mol is not None:
                        mg = self.featurizer(mol)
                        self._molgraph_cache[smi] = mg
                        molgraphs.append(mg)
                        atom_counts.append(mol.GetNumAtoms())
                    else:
                        molgraphs.append(None)
                        atom_counts.append(0)
                except Exception:
                    molgraphs.append(None)
                    atom_counts.append(0)
                continue

            # Slow path: 3-D bond inference (used when SMILES not available).
            mol_td = TensorDict({
                "coords": coords[i],
                "atomics": atomics[i],
                "padding_mask": padding_mask[i],
            })
            try:
                mol = self.converter.from_tensor(
                    mol_td,
                    rescale_coords=True,
                    sanitize=False,
                    use_openbabel=False,
                )
                if mol is not None:
                    mg = self.featurizer(mol)
                    molgraphs.append(mg)
                    atom_counts.append(mol.GetNumAtoms())
                else:
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


class CachedChemPropEncoder(MolecularEncoder):
    """ChemProp encoder backed by pre-computed CheMeleon embeddings.

    Since CheMeleon is frozen and 2-D (uses only molecular topology, not 3-D
    geometry), its output for a given SMILES is constant across all training
    steps, epochs, and conformers.  Pre-computing once and reading from an
    LMDB eliminates the GNN forward pass entirely during training, reducing
    encoder time to a fast lookup + padding operation.

    Usage
    -----
    1. Run playground/tabasco/chemeleon/precompute_embeddings.py once to build
       the embedding LMDB from all training SMILES.
    2. Point this encoder at that LMDB via the `lmdb_path` constructor arg.
    3. Swap ChemPropEncoder → CachedChemPropEncoder in the experiment config.

    For SMILES not present in the cache (e.g. validation molecules not seen
    during pre-computation) the encoder falls back to on-the-fly CheMeleon.
    """

    def __init__(self, lmdb_path: str, encoder_dim: int = 2048):
        """Args:
            lmdb_path: Path to the pre-computed embedding LMDB produced by
                playground/tabasco/chemeleon/precompute_embeddings.py.
            encoder_dim: Embedding dimensionality (must match the pre-compute
                run; default 2048 matches CheMeleon).
        """
        super().__init__()
        import lmdb as _lmdb

        self.encoder_dim = encoder_dim
        self._lmdb_path = lmdb_path
        self._db = _lmdb.open(
            lmdb_path,
            map_size=100 * (1024 ** 3),  # 100 GB map (virtual, not RSS)
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        # Fallback encoder for SMILES not found in the pre-computed cache.
        self._fallback = ChemPropEncoder(pretrained="chemeleon")
        # Freeze fallback weights too.
        for p in self._fallback.parameters():
            p.requires_grad = False

    def forward(self, coords, atomics, padding_mask, smiles=None):
        """Look up pre-computed embeddings; fall back to on-the-fly for misses.

        Args:
            coords: [B, N, 3]  (only used for fallback molecules)
            atomics: [B, N, atom_dim]  (only used for fallback molecules)
            padding_mask: [B, N]
            smiles: list of B SMILES strings (required for cache lookup)

        Returns:
            repr: [B, N, encoder_dim]
        """
        import pickle
        import numpy as np

        B, N, _ = coords.shape
        device = coords.device
        output = torch.zeros(B, N, self.encoder_dim, device=device)

        if smiles is None:
            # No SMILES — use fallback for entire batch.
            return self._fallback(coords, atomics, padding_mask)

        hit_indices, miss_indices = [], []
        with self._db.begin() as txn:
            for i, smi in enumerate(smiles):
                val = txn.get(smi.encode()) if smi else None
                if val is not None:
                    emb = pickle.loads(val)            # float16 [n_atoms, D]
                    emb_t = torch.from_numpy(emb.astype(np.float32)).to(device)
                    n = min(emb_t.shape[0], N)
                    output[i, :n] = emb_t[:n]
                    hit_indices.append(i)
                else:
                    miss_indices.append(i)

        if miss_indices:
            # Batch the misses and run the fallback encoder on them only.
            miss_coords   = coords[miss_indices]
            miss_atomics  = atomics[miss_indices]
            miss_padding  = padding_mask[miss_indices]
            miss_smiles   = [smiles[j] for j in miss_indices]
            with torch.no_grad():
                miss_repr = self._fallback(
                    miss_coords, miss_atomics, miss_padding, smiles=miss_smiles
                )
            for local_i, global_i in enumerate(miss_indices):
                output[global_i] = miss_repr[local_i]

        return output


class MACEEncoder(MolecularEncoder):
    """MACE-based 3D molecular encoder for REPA alignment.

    Uses MACE's equivariant neural network to extract atom-level embeddings
    from 3D molecular structures. Unlike ChemPropEncoder (2D graph-based),
    this encoder leverages 3D geometry and produces conformer-sensitive
    representations.

    MACE-OFF models are pretrained on molecular energies/forces and provide
    high-quality 3D-aware atom embeddings with zero sparsity (no ReLU).
    """

    # ATOM_NAMES from constants.py: [C, N, O, F, S, Cl, Br, I, *]
    # Map one-hot index -> atomic number for ASE
    ATOM_INDEX_TO_Z = {
        0: 6,   # C
        1: 7,   # N
        2: 8,   # O
        3: 9,   # F
        4: 16,  # S
        5: 17,  # Cl
        6: 35,  # Br
        7: 53,  # I
        8: 0,   # * (dummy — will be skipped)
    }

    def __init__(
        self,
        model_name: str = "small",
        invariants_only: bool = True,
        num_layers: int = -1,
        dataset_normalizer: float = 2.0,
    ):
        """Initialize MACE encoder.

        Args:
            model_name: MACE model size - "small", "medium", or "large"
            invariants_only: If True, extract only rotation-invariant features
            num_layers: Number of interaction layers to extract (-1 for all)
            dataset_normalizer: Coordinate scaling factor (coords * normalizer = Angstroms)
        """
        import os

        import numpy as np

        os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

        super().__init__()

        from e3nn import o3
        from mace.calculators import mace_off

        self.dataset_normalizer = dataset_normalizer

        # MACE sets torch.set_default_dtype(float64) globally during init.
        # Save and restore to avoid contaminating the rest of the process.
        _prev_dtype = torch.get_default_dtype()

        # Load pretrained MACE model; init on CPU, then cast to float32.
        # Lightning's .to(device) will move it to GPU when the trainer starts.
        self.calc = mace_off(model_name, device="cpu")
        self.mace_model = self.calc.models[0].float()
        self.model_name = model_name

        torch.set_default_dtype(_prev_dtype)

        # Freeze all MACE parameters
        for param in self.mace_model.parameters():
            param.requires_grad = False

        # Calculate encoder_dim from model architecture
        self.num_interactions = int(self.mace_model.num_interactions)
        self._num_layers = num_layers if num_layers != -1 else self.num_interactions
        self.invariants_only = invariants_only

        irreps_out = o3.Irreps(str(self.mace_model.products[0].linear.irreps_out))
        self.l_max = irreps_out.lmax
        self.num_invariant_features = irreps_out.dim // (self.l_max + 1) ** 2

        per_layer_features = [irreps_out.dim] * self.num_interactions
        per_layer_features[-1] = self.num_invariant_features
        self.encoder_dim = int(np.sum(per_layer_features[: self._num_layers]))

        # Note: mace.data, mace.tools.torch_geometric, and extract_invariant
        # are imported locally in forward() rather than stored as instance
        # attributes, because Python module objects are not picklable and
        # Lightning's save_hyperparameters() deepcopies the model.

    def forward(self, coords, atomics, padding_mask, smiles=None):
        """Extract MACE atom embeddings.

        Args:
            coords: [B, N, 3] - atomic coordinates (normalized by dataset_normalizer)
            atomics: [B, N, atom_dim] - one-hot atom types
            padding_mask: [B, N] - True for padding positions
            smiles: Ignored (accepted for interface compatibility)

        Returns:
            repr: [B, N, encoder_dim] - atom-level MACE representations
        """
        import ase
        from mace import data as mace_data
        from mace.modules.utils import extract_invariant
        from mace.tools import torch_geometric as mace_tg

        B, N, _ = coords.shape
        device = coords.device
        mace_device = next(self.mace_model.parameters()).device

        # Convert each molecule to ASE Atoms (requires CPU/numpy)
        atoms_list = []
        valid_indices = []
        atom_counts = []

        coords_cpu = coords.detach().cpu().numpy()
        atomics_cpu = atomics.detach().cpu()
        padding_cpu = padding_mask.detach().cpu()

        for i in range(B):
            mask = ~padding_cpu[i]
            n_atoms = mask.sum().item()

            if n_atoms == 0:
                continue

            # Convert one-hot to atomic numbers
            atom_indices = atomics_cpu[i][mask].argmax(dim=-1).numpy()
            atomic_numbers = [
                self.ATOM_INDEX_TO_Z.get(int(t), 6) for t in atom_indices
            ]

            # Skip dummy atoms (Z=0) — shouldn't happen for real atoms
            # but guard against padding leaking through
            real = [(z, j) for j, z in enumerate(atomic_numbers) if z > 0]
            if not real:
                continue
            atomic_numbers = [z for z, _ in real]
            real_idx = [j for _, j in real]

            # Denormalize coordinates back to Angstroms
            mol_coords = coords_cpu[i][mask.numpy()][real_idx] * self.dataset_normalizer

            atoms = ase.Atoms(numbers=atomic_numbers, positions=mol_coords)
            atoms_list.append(atoms)
            valid_indices.append(i)
            atom_counts.append(len(atomic_numbers))

        if len(atoms_list) == 0:
            return torch.zeros(B, N, self.encoder_dim, device=device, dtype=torch.float32)

        # Build MACE AtomicData objects. MACE's data pipeline reads
        # torch.get_default_dtype(), so we temporarily set float32 to match
        # the model weights (cast from float64 at init).
        _prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        try:
            keyspec = mace_data.KeySpecification(
                info_keys=self.calc.info_keys, arrays_keys=self.calc.arrays_keys
            )
            configs = [
                mace_data.config_from_atoms(
                    x, key_specification=keyspec, head_name=self.calc.head
                )
                for x in atoms_list
            ]
            dataset = [
                mace_data.AtomicData.from_config(
                    config,
                    z_table=self.calc.z_table,
                    cutoff=self.calc.r_max,
                    heads=self.calc.available_heads,
                )
                for config in configs
            ]

            loader = mace_tg.dataloader.DataLoader(
                dataset=dataset,
                batch_size=len(dataset),
                shuffle=False,
                drop_last=False,
            )
            batch = next(iter(loader)).to(mace_device)

            with torch.no_grad():
                output = self.mace_model(batch.to_dict(), compute_force=False)
                node_feats = output["node_feats"]

                if self.invariants_only:
                    descriptors = extract_invariant(
                        node_feats,
                        num_layers=self._num_layers,
                        num_features=self.num_invariant_features,
                        l_max=self.l_max,
                    )
                else:
                    descriptors = node_feats

                descriptors = descriptors[:, : self.encoder_dim].float()
        finally:
            torch.set_default_dtype(_prev_dtype)

        # Reconstruct [B, N, encoder_dim] with padding
        output_tensor = torch.zeros(B, N, self.encoder_dim, device=device, dtype=torch.float32)
        batch_indices = batch["batch"]

        for local_idx, global_idx in enumerate(valid_indices):
            atom_mask = batch_indices == local_idx
            n_atoms = atom_counts[local_idx]
            output_tensor[global_idx, :n_atoms] = descriptors[atom_mask]

        return output_tensor


class Projector(nn.Module):
    """Projects diffusion hidden states to encoder embedding space.

    This is a trainable projection layer that maps the diffusion model's
    hidden states to the same dimensionality as the frozen encoder's
    embeddings, enabling alignment via cosine similarity or MSE loss.

    The first layer uses ``nn.LazyLinear`` so the input dimension is
    inferred automatically on the first forward pass.  This means the
    projector works regardless of how many hidden-state heads are fused
    (e.g. ``cross_attention=True`` concatenates coord + atom heads).
    """

    def __init__(self, hidden_dim: int, encoder_dim: int, num_layers: int = 2):
        """Args:
        hidden_dim: Intermediate MLP width
        encoder_dim: Dimension of encoder embeddings (output size)
        num_layers: Number of MLP layers
        """
        super().__init__()

        layers = []
        if num_layers == 1:
            layers.append(nn.LazyLinear(encoder_dim))
        else:
            layers.extend([nn.LazyLinear(hidden_dim), nn.SiLU()])
            for _ in range(num_layers - 2):
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
            layers.append(nn.Linear(hidden_dim, encoder_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, hidden_states):
        """Project hidden states to encoder space.

        Args:
            hidden_states: [B, N, hidden_dim]

        Returns:
            projected: [B, N, encoder_dim]
        """
        return self.mlp(hidden_states)
