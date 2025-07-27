from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from common.entity import CaseID, FeatureName, OrganName


class AbsolutePositionAndDGDataset(Dataset):
    def __init__(
        self,
        paths: List[Path],
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index) -> Dict[str, Union[torch.Tensor, str, int]]:
        path = self.paths[index]
        case_id, organ_name, feature_name = self.__extract_info(path)
        feature = np.load(path)
        feature = torch.from_numpy(feature).to(torch.float32)
        if self.transform:
            feature = self.transform(feature)
        return {
            "data": feature,
            "path": str(path),
            "case_id": case_id.value,
            "organ_name": organ_name.value,
            "feature_name": feature_name.value,
        }

    def __extract_info(self, path: Path) -> Tuple[CaseID, OrganName, FeatureName]:
        file_name = path.stem
        info = file_name.split("_")
        return CaseID(info[0]), OrganName(info[1]), FeatureName(info[2])


class AbsolutePositionAndDGFeatureDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_paths: List[Path],
        val_paths: List[Path],
        test_paths: List[Path],
        batch_size=32,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = AbsolutePositionAndDGDataset(
            self.hparams.train_paths, transform=self.hparams.transform
        )
        self.val_dataset = AbsolutePositionAndDGDataset(self.hparams.val_paths)
        self.test_dataset = AbsolutePositionAndDGDataset(self.hparams.test_paths)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size)
