from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from common.consts import NORMALIZE_RATIO
from common.entity import CaseID, FeatureName, OrganName
from src.mesh.mesh import OrganMesh


class PositionalDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        organ_name: OrganName,
        case_ids: List[CaseID],
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self.organ_name = organ_name
        self.transform = transform
        self.paths: list[Path] = []
        self.case_ids: list[CaseID] = []

        for case_id in case_ids:
            mesh_path = (
                data_dir / case_id.value / "00/output" / f"{organ_name.value}.ply"
            )
            if not mesh_path.exists():
                continue

            self.paths.append(mesh_path)
            self.case_ids.append(case_id)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index) -> Dict[str, Union[torch.Tensor, str, int]]:
        path = self.paths[index]
        case_id = self.case_ids[index]
        mesh = OrganMesh(path)
        feature = (
            torch.from_numpy(mesh.nodes.flatten()).to(torch.float32) / NORMALIZE_RATIO
        )
        if self.transform:
            feature = self.transform(feature)
        return {
            "data": feature,
            "path": str(path),
            "case_id": case_id.value,
            "organ_name": self.organ_name.value,
            "feature_name": FeatureName.Position.value,
        }


class PositionalFeatureDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        train_case_ids: List[CaseID],
        val_case_ids: List[CaseID],
        test_case_ids: List[CaseID],
        organ_name: OrganName,
        batch_size=32,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = PositionalDataset(
            self.hparams.data_dir,
            case_ids=self.hparams.train_case_ids,
            organ_name=self.hparams.organ_name,
            transform=self.hparams.transform,
        )
        self.val_dataset = PositionalDataset(
            self.hparams.data_dir,
            case_ids=self.hparams.val_case_ids,
            organ_name=self.hparams.organ_name,
        )
        self.test_dataset = PositionalDataset(
            self.hparams.data_dir,
            case_ids=self.hparams.test_case_ids,
            organ_name=self.hparams.organ_name,
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size)
