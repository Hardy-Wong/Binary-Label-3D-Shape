import io
import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import List

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from torch import nn

import wandb
from src.common.entity import CaseID, ModelName
from src.common.splited_data import load_case_ids
from src.mesh.mesh import OrganMesh
from src.models.AbsolutePositionAndDGMeshLinearVAE.model_AbsolutePositionAndDGMeshLinearVAE import (
    AbsolutePositionAndDGMeshLinearVAE,
)
from src.models.AbsolutePositionMeshLinearVAE.data_AbsolutePositionAndDGMeshLinearVAE import (
    PositionalFeatureDataModule,
)
from src.models.AbsolutePositionMeshLinearVAE.model_AbsolutePositionMeshLinearVAE import (
    AbsolutePositionMeshLinearVAE,
)
from src.common.consts import NORMALIZE_RATIO
from src.common.entity import FeatureName, OrganName
from src.models.AbsolutePositionAndDGMeshLinearVAE.data_AbsolutePositionAndDGMeshLinearVAE import (
    AbsolutePositionAndDGFeatureDataModule,
)
from src.models.augment import PositionalDataAugmentation
from src.visualize.organ import draw_organ

log = logging.getLogger(__name__)


def model_factory(model_name: ModelName, cfg: OmegaConf):
    if model_name == ModelName.absolute_position_mesh_linear_VAE:
        input_dim, output_dim = estimate_input_output_dim(
            Path(cfg.template_mesh_path), model_name=model_name
        )
        return AbsolutePositionMeshLinearVAE(
            input_dim=input_dim,
            output_dim=output_dim,
            latent_dim=cfg.latent_dim,
            hidden_dims=cfg.hidden_dims,
            sigma_max=2.0,
        )
    elif model_name == ModelName.absolute_position_and_dg_mesh_linear_VAE:
        input_dim, output_dim = estimate_input_output_dim(
            Path(cfg.template_mesh_path), model_name=model_name
        )
        return AbsolutePositionAndDGMeshLinearVAE(
            input_dim=input_dim,
            output_dim=output_dim,
            latent_dim=cfg.latent_dim,
            hidden_dims=cfg.hidden_dims,
            sigma_max=2.0,
            output_only_position=False,
        )
    else:
        raise ValueError


def data_factory(feature_name: FeatureName, cfg: OmegaConf):
    if feature_name == FeatureName.DGandPosition:
        data_dir = Path(cfg.data_dir)
        if cfg.debug_mode:
            file_names = list(data_dir.glob("*.npy"))
            file_names = file_names[:20]
            train_file_names, rest_file_names = train_test_split(
                file_names, test_size=0.3, train_size=0.7
            )
            test_file_names, val_file_names = train_test_split(
                rest_file_names, test_size=0.5, train_size=0.5
            )
        else:

            def file_name(idx):
                return f"{idx}_liver_DGandPosition.npy"

            train_file_names = [
                file_name(case_id.value) for case_id in load_case_ids(mode="train")
            ]
            test_file_names = [
                file_name(case_id.value) for case_id in load_case_ids(mode="test")
            ]
            val_file_names = [
                file_name(case_id.value) for case_id in load_case_ids(mode="val")
            ]
        log.info(f"train_file_names: {train_file_names}")
        log.info(f"train_file_num: {len(train_file_names)}")
        log.info(f"test_file_names: {test_file_names}")
        log.info(f"test_file_num: {len(test_file_names)}")
        log.info(f"val_file_names: {val_file_names}")
        log.info(f"val_file_num: {len(val_file_names)}")
        train_file_paths = [data_dir / file_name for file_name in train_file_names]
        val_file_paths = [data_dir / file_name for file_name in val_file_names]
        test_file_paths = [data_dir / file_name for file_name in test_file_names]

        return AbsolutePositionAndDGFeatureDataModule(
            train_paths=train_file_paths,
            val_paths=val_file_paths,
            test_paths=test_file_paths,
            transform=PositionalDataAugmentation(ratio=cfg.augmentation.moving_ratio),
        )
    elif feature_name.Position:
        data_dir = Path(cfg.data_dir)
        if cfg.debug_mode:
            case_ids = []
            for child in data_dir.iterdir():
                if child.is_dir:
                    try:
                        case_ids.append(CaseID(child.name))
                    except AssertionError:
                        continue
                if len(case_ids) == 20:
                    break
            train_case_ids, rest_case_ids = train_test_split(
                case_ids, test_size=0.3, train_size=0.7
            )
            test_case_ids, val_case_ids = train_test_split(
                rest_case_ids, test_size=0.5, train_size=0.5
            )
        else:
            train_case_ids = [
                CaseID(case_id.value) for case_id in load_case_ids(mode="train")
            ]
            test_case_ids = [
                CaseID(case_id.value) for case_id in load_case_ids(mode="test")
            ]
            val_case_ids = [
                CaseID(case_id.value) for case_id in load_case_ids(mode="val")
            ]
        log.info(f"train_case_ids: {train_case_ids}")
        log.info(f"train_case_num: {len(train_case_ids)}")
        log.info(f"test_case_ids: {test_case_ids}")
        log.info(f"test_case_num: {len(test_case_ids)}")
        log.info(f"val_case_ids: {val_case_ids}")
        log.info(f"val_case_num: {len(val_case_ids)}")

        return PositionalFeatureDataModule(
            data_dir=Path(cfg.data_dir),
            organ_name=OrganName.liver,
            train_case_ids=train_case_ids,
            val_case_ids=val_case_ids,
            test_case_ids=test_case_ids,
            transform=PositionalDataAugmentation(ratio=cfg.augmentation.moving_ratio),
        )


class MeshVAELTModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        recon_weight: float = 1.0,
        print_summary=False,
    ) -> None:
        super(MeshVAELTModule, self).__init__()
        self.lr = lr
        self.model = model
        self.recon_weight = recon_weight
        # TODO: fix below configure
        self.logging_image_frec = 100
        self.original_data_dir = Path("/work/3D-CT PLY")

        # if print_summary:
        #     from torchsummary import summary
        #     log.info('\n' + str(summary(self.model, input_size=(32, 6000))))

    def forward(self, x):
        return self.model(x["data"])

    def training_step(self, batch, batch_idx):
        preds = self.forward(batch)
        losses = self.model.loss_function(*preds, M_N=self.recon_weight)
        for k in losses.keys():
            self.log(
                f"train_{ k }", losses[k], prog_bar=True, on_epoch=True, on_step=False
            )
        log.info(f"train_loss: {losses['loss']}")
        return losses

    def validation_step(self, batch, batch_idx):
        preds = self.forward(batch)
        losses = self.model.loss_function(*preds, M_N=self.recon_weight)
        for k in losses.keys():
            self.log(
                f"val_{ k }", losses[k], prog_bar=True, on_epoch=True, on_step=False
            )
        log.info(f"val_loss: {losses['loss']}")

        if self.current_epoch % self.logging_image_frec == 0:
            case_ids = []
            gt_meshes = []
            estimated_meshes = []
            organ_name = batch["organ_name"][0]
            for i, case_id in enumerate(batch["case_id"][:3]):
                case_id = CaseID(case_id)
                original_mesh = OrganMesh(
                    self.original_data_dir
                    / case_id.value
                    / "00/output"
                    / f"{organ_name}.ply"
                )
                tmp_mesh = deepcopy(original_mesh)
                tmp_mesh.nodes = (
                    preds[0][i].to("cpu").detach().numpy().copy().reshape((-1, 3))
                    * NORMALIZE_RATIO
                )
                case_ids.append(case_id)
                gt_meshes.append(original_mesh)
                estimated_meshes.append(tmp_mesh)

            intermediate_result_img = self.__visualize(
                gt_meshes=gt_meshes,
                estimated_meshes=estimated_meshes,
                case_ids=case_ids,
            )
            self.logger.log_image("val_result", [intermediate_result_img])
        return losses

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def __visualize(
        self,
        gt_meshes: List[OrganMesh],
        estimated_meshes: List[OrganMesh],
        case_ids: List[CaseID],
    ) -> Image:
        assert len(gt_meshes) == len(estimated_meshes) == len(case_ids)
        fig, axs = plt.subplots(
            nrows=len(case_ids),
            ncols=2,
            figsize=(2 * 3, len(case_ids) * 3),
            subplot_kw=dict(projection="3d"),
        )
        plt.title("Ground Trueth <----> Estimated Result")
        for i, (ax_left, ax_right) in enumerate(axs):
            ax_left.set_title(case_ids[i].value, loc="left")
            draw_organ(fig, ax_left, gt_meshes[i], marker="o")
            draw_organ(fig, ax_right, estimated_meshes[i], marker="o")
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        return Image.open(buf)


def estimate_input_output_dim(template_mesh_path: Path, model_name: ModelName) -> int:
    template = OrganMesh(template_mesh_path)
    num_nodes = template.num_nodes
    num_edges = (
        sum([len(template.get_vertex_adjacent_vertices(i)) for i in range(num_nodes)])
        / 2
    )  # remove double count
    if model_name == ModelName.mesh_VAE:
        return int(num_nodes * 6 + num_edges * 3 * 2), int(
            num_nodes * 6 + num_edges * 3 * 2
        )
    elif model_name == ModelName.absolute_position_and_dg_mesh_linear_VAE:
        return int(12 * num_nodes), num_nodes * 3
    elif model_name == ModelName.absolute_position_mesh_linear_VAE:
        return num_nodes * 3, num_nodes * 3
    else:
        raise ValueError("You should specify correct name of model.")


def create_latent_vecs_after_train_script(
    model: MeshVAELTModule, data: AbsolutePositionAndDGFeatureDataModule
):
    """create latent vectors

    Args:
        model (pl.LightningModule): trained model
        data (pl.LightningDataModule): data_module

    Raises:
        ValueError: _description_
    """
    log.info("create_latent_vecs....")
    model.freeze()
    latent_save_root_dir = Path("latent_vecs")
    os.makedirs(latent_save_root_dir, exist_ok=True)
    for mode in ["train", "test", "val"]:
        if mode == "train":
            datas = iter(data.train_dataloader())
        elif mode == "test":
            datas = iter(data.test_dataloader())
        else:
            datas = iter(data.val_dataloader())
        latent_save_dir = latent_save_root_dir / mode
        os.makedirs(latent_save_dir, exist_ok=True)
        for batch in datas:
            for d, path in zip(batch["data"], batch["path"]):
                encoded = model.model.encode(torch.reshape(d, (1, -1)))
                z = model.model.reparameterize(*encoded)
                case_id, organ, _ = path.split("/")[-1].split("_")  # TODO: 関数に切り出す
                np.save(
                    str(latent_save_dir / f"{case_id}_{organ}_latent_vec.npy"),
                    z.to("cpu").detach().numpy(),
                )


@hydra.main(
    version_base="1.1", config_path="/work/src/conf", config_name="train_mesh_VAE"
)
def main(cfg: DictConfig) -> None:
    log.info(f"job_id: {cfg.job_id}")
    if cfg.commit_id == "none-select":
        raise ValueError(
            "commit id should be specified! please run GIT_COMMIT_HASH=$(git rev-parse HEAD)"
        )
    seed = cfg.random_seed
    pl.seed_everything(seed=seed)

    wandb.finish()
    # import datetime
    # now = datetime.datetime.now()
    # time_stamp = now.strftime("%Y/%m/%d_%H:%M:%S")
    project_name = cfg.project_name
    experiment_name = (
        cfg.experiment_name + " moving_ratio=" + str(cfg.augmentation.moving_ratio)
    )
    run = wandb.init(project=project_name, name=experiment_name)
    wandb_logger = WandbLogger(project=project_name, name=experiment_name)
    data_module = data_factory(feature_name=FeatureName.Position, cfg=cfg)
    model = model_factory(
        model_name=ModelName.absolute_position_mesh_linear_VAE, cfg=cfg
    )
    lt_module = MeshVAELTModule(
        model=model, recon_weight=cfg.recon_weight, print_summary=cfg.print_summary
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        dirpath="checkpoint_model",
        filename="mesh_VAE_{epoch:02d}-{val_loss:.2f}",
    )
    trainer = pl.Trainer(
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=50),
            checkpoint_callback,
        ],
        logger=wandb_logger,
    )
    trainer = pl.Trainer(max_epochs=300, logger=wandb_logger)
    trainer.fit(lt_module, datamodule=data_module)
    log.info(f"wandb: summary \n {run.summary()}")

    create_latent_vecs_after_train_script(lt_module, data_module)


if __name__ == "__main__":
    main()
