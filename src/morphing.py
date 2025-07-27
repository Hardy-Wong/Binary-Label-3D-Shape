from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from common.entity import ModelName
from src.mesh.mesh import OrganMesh
from train_mesh_VAE import MeshVAELTModule, model_factory


@hydra.main(version_base="1.1", config_path="/work/src/conf", config_name="morphing")
def main(cfg: DictConfig):
    model = model_factory(
        model_name=ModelName.absolute_position_mesh_linear_VAE, cfg=cfg
    )
    model = MeshVAELTModule.load_from_checkpoint(cfg.checkpoint_path, model=model)
    model.freeze()
    model.to("cpu")
    feature_A = (
        torch.from_numpy(np.load(cfg.feature_A_path))
        .to(torch.float32)
        .to(model.device)[None, :]
    )
    feature_B = (
        torch.from_numpy(np.load(cfg.feature_B_path))
        .to(torch.float32)
        .to(model.device)[None, :]
    )

    encoded_A = model.model.encode(feature_A)
    z_A = model.model.reparameterize(*encoded_A)
    encoded_B = model.model.encode(feature_B)
    z_B = model.model.reparameterize(*encoded_B)
    for i in tqdm(range(cfg.interpolate_num)):
        z = z_A * (1.0 - i / cfg.interpolate_num) + z_B * (i / cfg.interpolate_num)
        decoded_nodes = model.model.decode(z)[0]
        tmp = OrganMesh(Path(cfg.template_mesh_path))
        tmp.nodes = (
            decoded_nodes.to("cpu").detach().numpy().copy().reshape((-1, 3)) * 1e3
        )
        tmp.save(f"{i}.ply")


if __name__ == "__main__":
    main()
