from argparse import Namespace
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from lightning_fabric.loggers import Logger as BaseLogger
from lightning_fabric.loggers.logger import _DummyExperiment

from common.entity import FeatureName, ModelName, OrganName, ModelProvider
from models.factories.callback_factory import callback_factory
from models.factories.data_factory import data_factory
from models.factories.model_factory import model_factory
from models.factories.latent_logger_factory import latent_logger_factory
from post_process import post_process
from mesh.mesh import OrganMesh


log = logging.getLogger(__name__)


class LightningFileLogger(BaseLogger):
    def __init__(self, logger: logging.Logger) -> None:
        super().__init__()
        self.logger = logger

    @property
    def name(self) -> Optional[str]:
        return "LightningFileLogger"
    @property
    def version(self) -> Union[int, str, None]:
        return "0.1.0"

    def after_save_checkpoint(self, checkpoint_callback) -> None:
        pass

    @property
    def save_dir(self) -> Optional[str]:
        return None

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        self.logger.info(f"step: {step}  metrics: {metrics}")
        return None

    def log_hyperparams(
        self, params: Union[Dict[str, Any], Namespace], *args: Any, **kwargs: Any
    ) -> None:
        self.logger.info(params)

    def experiment(self):
        return _DummyExperiment()


@hydra.main(
    version_base="1.1", config_path="conf", config_name="train_mesh_VAE"
)
def main(cfg: DictConfig) -> None:
    seed = cfg.random_seed
    pl.seed_everything(seed=seed)
    # wandb.finish()
    project_name = cfg.project_name
    experiment_name = cfg.experiment_name
    model_name = ModelName[cfg.model_name]
    experiment_id = cfg.commit_hash
    target_organ = OrganName[cfg.target_organ]
    template_mesh = OrganMesh(cfg.template_mesh_path)

    # run = wandb.init(project=project_name, name=experiment_name)
    # wandb_logger = WandbLogger(project=project_name, name=experiment_name)
    # factoryメソッドで各種モジュールを引っ張ってくる。
    data_module = data_factory(
        feature_name=FeatureName[cfg.feature_name],
        model_provider=ModelProvider[cfg.model_provider],
        cfg=cfg,
        target_organ=target_organ,
        log=log,
    )
    data_module.setup(stage="train")
    sample_data = data_module.train_dataset.__getitem__(0)
    if cfg.model_provider == ModelProvider.PyG.value:
        log.info(f'data shape: {sample_data["data"]["x"].shape}')
    elif cfg.model_provider == ModelProvider.torch.value:
        log.info(f"data shape: {sample_data['data'].shape}")
    callbacks = callback_factory(cfg.debug_mode)
    latent_logger = latent_logger_factory(
        model_name=model_name,
        experiment_id=experiment_id,
        number_of_vertices=template_mesh.num_nodes,
    )
    lt_module = model_factory(
        model_name=model_name,
        cfg=cfg,
        latent_logger=latent_logger,
        commit_hash=experiment_id,
        normal_logger=log,
        organ_name=target_organ,
    )

    # 学習
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0], 
        max_epochs=1 if cfg.debug_mode else cfg.max_epochs,
        callbacks=callbacks,
        # logger=[wandb_logger, LightningFileLogger(log)],
        logger=[ LightningFileLogger(log)],
    )
    trainer.fit(lt_module, datamodule=data_module)

    # 推論
    trainer.predict(
        lt_module, dataloaders=data_module.train_dataloader(), ckpt_path="best"
    )
    trainer.predict(
        lt_module, dataloaders=data_module.val_dataloader(), ckpt_path="best"
    )
    trainer.predict(
        lt_module, dataloaders=data_module.test_dataloader(), ckpt_path="best"
    )
    # log.info(f"wandb: summary \n {dict(run.summary)}")
    post_process(
        log_vertex_position_df_dir=lt_module.mesh_vertices_position_diff_logger.log_dir.resolve(
            strict=True
        ),
        commit_hash=experiment_id,
        template_mesh_path=Path(cfg.template_mesh_path),
        organ_name=target_organ,
        will_plot_metrics=cfg.will_plot_metrics,
        will_plot_samples=cfg.will_plot_samples,
    )


if __name__ == "__main__":
    main()
