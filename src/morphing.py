from pathlib import Path

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from mesh.mesh import OrganMesh


@hydra.main(version_base="1.1", config_path="conf", config_name="morphing")
def main(cfg: DictConfig):
    if cfg.mode == "morphing":
        morphing(cfg)
    elif cfg.mode == "linear_interpolate":
        linear_interpolate(cfg)
    else:
        raise NotImplementedError(f"invalid mode: {cfg.mode}")


def morphing(cfg: DictConfig):
    raise DeprecationWarning("morphing is not implemented yet.")
    # model = model_factory(model_name=ModelName(cfg.model_name), cfg=cfg)
    # model = MeshVAELTModule.load_from_checkpoint(cfg.checkpoint_path, model=model)
    # model.freeze()
    # model.to("cpu")
    # z_A = (
    #     torch.from_numpy(np.load(cfg.latent_A_path))
    #     .to(torch.float32)
    #     .to(model.device)[None, :]
    # )
    # z_B = (
    #     torch.from_numpy(np.load(cfg.latent_B_path))
    #     .to(torch.float32)
    #     .to(model.device)[None, :]
    # )
    # if ModelProvider[cfg.model_provider] == ModelProvider.PyG:
    #     template_mesh = OrganMesh(Path(cfg.template_mesh_path))
    #     edge_index = cal_edge_index(template_mesh=template_mesh)
    #     for i in tqdm(range(cfg.interpolate_num)):
    #         z = z_A * (1.0 - i / cfg.interpolate_num) + z_B * (i / cfg.interpolate_num)
    #         decoded_nodes = model.model.decode(z, edge_index)[0]
    #         decoded_nodes = select_positions(
    #             decoded_nodes, num_of_f_in_1_node=12, model_provider=ModelProvider.PyG
    #         )
    #         tmp = OrganMesh(Path(cfg.template_mesh_path))
    #         tmp.nodes = (
    #             decoded_nodes.to("cpu").detach().numpy().copy().reshape((-1, 3))
    #             * POSITIONAL_NORMALIZE_RATIO
    #         )
    #         print(tmp.nodes[0])
    #         tmp.save(f"{i}.ply")
    # elif ModelProvider[cfg.model_provider] == ModelProvider.torch:
    #     for i in tqdm(range(cfg.interpolate_num)):
    #         z = z_A * (1.0 - i / cfg.interpolate_num) + z_B * (i / cfg.interpolate_num)
    #         decoded_nodes = model.model.decode(z)[0]
    #         tmp = OrganMesh(Path(cfg.template_mesh_path))
    #         tmp.nodes = (
    #             decoded_nodes.to("cpu").detach().numpy().copy().reshape((-1, 3))
    #             * POSITIONAL_NORMALIZE_RATIO
    #         )
    #         tmp.save(f"{i}.ply")
    # else:
    #     raise ValueError


def linear_interpolate(cfg: DictConfig):
    mesh_A = OrganMesh(Path(cfg.mesh_A_path))
    mesh_B = OrganMesh(Path(cfg.mesh_B_path))
    tmp = OrganMesh(Path(cfg.mesh_A_path))
    for i in tqdm(range(cfg.interpolate_num + 1)):
        tmp.nodes = mesh_A.nodes * (1.0 - i / cfg.interpolate_num) + mesh_B.nodes * (
            i / cfg.interpolate_num
        )
        tmp.save(f"{i}.ply")


if __name__ == "__main__":
    main()
