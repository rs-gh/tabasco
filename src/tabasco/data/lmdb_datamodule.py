import os
from typing import Optional

from lightning import LightningDataModule
from tabasco.data.utils import TensorDictCollator
from tabasco.data.components.lmdb_unconditional import UnconditionalLMDBDataset
from torch.utils.data import DataLoader
from tabasco.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

# HuggingFace dataset mappings for auto-download
HF_DATASETS = {
    "qm9": {
        "repo_id": "carlosinator/tabasco-qm9",
        "files": {
            "train": "processed_qm9_train.pt",
            "val": "processed_qm9_val.pt",
            "test": "processed_qm9_test.pt",
        },
    },
    "geom": {
        "repo_id": "carlosinator/tabasco-geom-drugs",
        "files": {
            "train": "processed_geom_train.pt",
            "val": "processed_geom_val.pt",
            "test": "processed_geom_test.pt",
        },
    },
}


class LmdbDataModule(LightningDataModule):
    """PyTorch Lightning `DataModule` for unconditional ligand generation."""

    def __init__(
        self,
        data_dir: str,
        lmdb_dir: str,
        add_random_rotation: bool = False,
        add_random_permutation: bool = False,
        reorder_to_smiles_order: bool = False,
        remove_hydrogens: bool = True,
        batch_size: int = 256,
        num_workers: int = 0,
        val_data_dir: Optional[str] = None,
        test_data_dir: Optional[str] = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.val_data_dir = val_data_dir
        self.test_data_dir = test_data_dir
        self.lmdb_dir = lmdb_dir
        self.dataset_kwargs = {
            "add_random_rotation": add_random_rotation,
            "add_random_permutation": add_random_permutation,
            "reorder_to_smiles_order": reorder_to_smiles_order,
            "remove_hydrogens": remove_hydrogens,
        }
        """Args:
            data_dir: Path to the training set .pt file produced by preprocessing.
            lmdb_dir: Directory where LMDB files and stats are stored.
            add_random_rotation: Apply random rotations inside each dataset item.
            add_random_permutation: Randomly permute heavy-atom order in each item.
            reorder_to_smiles_order: Re-index atoms to canonical SMILES before tensorization.
            remove_hydrogens: Strip explicit hydrogens before conversion to tensors.
            batch_size: Number of molecules per batch.
            num_workers: DataLoader worker count.
            val_data_dir: Optional path to a separate validation set; if None, a train/val split is created.
            test_data_dir: Optional path to a held-out test set.
        """

    def prepare_data(self):
        """Download data from HuggingFace if missing, create LMDB directory."""
        # Detect dataset type from path
        dataset_type = None
        for dtype in HF_DATASETS:
            if dtype in self.data_dir.lower():
                dataset_type = dtype
                break

        if dataset_type is None:
            log.info("Unknown dataset type, skipping auto-download")
            return

        hf_config = HF_DATASETS[dataset_type]

        # Check and download each split if missing
        splits_to_check = {
            "train": self.data_dir,
            "val": self.val_data_dir,
            "test": self.test_data_dir,
        }

        for split_name, file_path in splits_to_check.items():
            if file_path and not os.path.exists(file_path):
                log.info(f"Data file missing: {file_path}")
                log.info(f"Downloading {split_name} split from HuggingFace...")

                try:
                    from huggingface_hub import hf_hub_download

                    # Ensure parent directory exists
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)

                    # Download from HuggingFace
                    hf_hub_download(
                        repo_id=hf_config["repo_id"],
                        filename=hf_config["files"][split_name],
                        local_dir=os.path.dirname(file_path),
                        repo_type="dataset",
                    )
                    log.info(f"Downloaded {split_name} split to {file_path}")
                except Exception as e:
                    log.warning(f"Failed to download {split_name}: {e}")
                    log.warning(
                        f"Please manually download from https://huggingface.co/datasets/{hf_config['repo_id']}"
                    )

        # Create LMDB directory if it doesn't exist
        if self.lmdb_dir and not os.path.exists(self.lmdb_dir):
            os.makedirs(self.lmdb_dir, exist_ok=True)
            log.info(f"Created LMDB directory: {self.lmdb_dir}")

    def setup(self, stage: Optional[str] = None):
        """Instantiate train/val/test datasets.

        If val_data_dir is None, the training file is randomly split into
        train and validation indices. Otherwise the provided paths are used.
        """
        if self.val_data_dir is None:
            train_indices, val_indices = self._compute_train_val_split()  # nosec B614

            log.info("Initializing train dataset...")
            self.train_dataset = UnconditionalLMDBDataset(
                data_dir=self.data_dir,
                split_indices=train_indices,
                **self.dataset_kwargs,
            )
            log.info(
                f"Train dataset initialized with {len(self.train_dataset)} samples"
            )

            log.info("Initializing val dataset...")
            self.val_dataset = UnconditionalLMDBDataset(
                data_dir=self.data_dir,
                split_indices=val_indices,
                **self.dataset_kwargs,
            )
            log.info(f"Val dataset initialized with {len(self.val_dataset)} samples")

        else:
            self.train_dataset = UnconditionalLMDBDataset(
                data_dir=self.data_dir,
                split="train",
                lmdb_dir=self.lmdb_dir,
                **self.dataset_kwargs,
            )
            self.val_dataset = UnconditionalLMDBDataset(
                data_dir=self.val_data_dir,
                split="val",
                lmdb_dir=self.lmdb_dir,
                **self.dataset_kwargs,
            )
            if self.test_data_dir is not None:
                self.test_dataset = UnconditionalLMDBDataset(
                    data_dir=self.test_data_dir,
                    split="test",
                    lmdb_dir=self.lmdb_dir,
                    **self.dataset_kwargs,
                )
            else:
                self.test_dataset = UnconditionalLMDBDataset(
                    data_dir=self.val_data_dir,
                    split="val",
                    lmdb_dir=self.lmdb_dir,
                    **self.dataset_kwargs,
                )

    def get_dataset_stats(self):
        """Return statistics dictionary computed by the training dataset."""
        return self.train_dataset.get_stats()

    def train_dataloader(self):
        """Return the training `DataLoader`."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=TensorDictCollator(),
            shuffle=True,
        )

    def val_dataloader(self):
        """Return the validation `DataLoader`."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=TensorDictCollator(),
            shuffle=False,
        )

    def test_dataloader(self):
        """Return the test `DataLoader` (falls back to validation set when absent)."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=TensorDictCollator(),
            shuffle=False,
        )
