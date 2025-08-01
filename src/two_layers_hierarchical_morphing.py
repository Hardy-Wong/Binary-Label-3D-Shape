from os import listdir, environ
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import hydra
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from omegaconf import DictConfig
import torch

from common.entity import ModelName, CaseID
from mesh.mesh import OrganMesh
from common.entity import OrganName
from common.consts import POSITIONAL_NORMALIZE_RATIO
from models.factories.model_factory import model_factory
from models.factories.latent_logger_factory import latent_logger_factory
from common.splited_data import load_case_ids
import logging
from tqdm import tqdm


log = logging.getLogger(__name__)


def load_zs(data_dir: Path, case_id: CaseID, organ_name: OrganName):
    file_name = f"_{case_id}_{organ_name.name}_latent_vec.npy"
    z1 = torch.Tensor(np.load(data_dir / "layer_1" / file_name))
    z2 = torch.Tensor(np.load(data_dir / "layer_2" / file_name))
    return z1, z2

def load_pca_weight(weight_dir: Path, case_id: CaseID) -> np.ndarray:
    with open(weight_dir / f"{case_id}.pkl", "rb") as f:
        study = pickle.load(f)
    return np.array([study.best_params[f"weight_{i}"] for i in range(len(study.best_params))])

def pca_reconstruct(weight: list[float], pca: PCA, mean_mesh: np.ndarray):
    eigen_vectors = pca.components_
    non_centered_result = (weight @ eigen_vectors).reshape(mean_mesh.shape) * POSITIONAL_NORMALIZE_RATIO
    result = mean_mesh + non_centered_result
    return result



def find_ckpt(exp_dir: Path):
    search_result = exp_dir.glob("**/*.ckpt")
    search_result = list(search_result)
    if len(search_result) == 0:
        raise FileNotFoundError("There is no ckpt file.")
    return search_result[0]


def load_organ_nodes_dataframe(
    organ_name: OrganName,
    data_dir=Path("/home/snail/mesh_gcn_vae/data/3D-CT PLY"),
    emit: bool = False,
) -> pd.DataFrame:
    vs = []
    vertex_ids, case_ids, organ_names = [], [], []
    for dire in tqdm(listdir(data_dir)):
        organ_path = data_dir / dire / "00/output" / f"{organ_name.value}.ply"
        if not organ_path.exists():
            continue
        mesh = OrganMesh(organ_path)
        vs.append(mesh.nodes.flatten())
        case_ids.append(dire)

    if emit:
        vs_df = pd.DataFrame(vs)
        vs_df.index = case_ids
        return vs_df
    else:
        return pd.DataFrame(
            data={
                "vs": vs,
                "case_id": case_ids,
            }
        )


@hydra.main(
    version_base="1.1",
    config_path=str(Path(environ["WORKING_DIR"]) / "src" / "conf"),
    config_name="two_layers_hierarchical_morphing.yaml",
)
def main(cfg: DictConfig):
    print("process start")
    if cfg.mode == "hmvae_morphing":
        hmvae_morphing(cfg)
    elif cfg.mode == "pca_morphing":
        pca_morphing(cfg)
    else:
        raise NotImplementedError(f"mode {cfg.mode} is not implemented.")


def pca_morphing(cfg: DictConfig):
    df = load_organ_nodes_dataframe(
        organ_name=OrganName(cfg.target_organ),
        emit=True,
    )
    train_ids = load_case_ids(mode="train")
    train_ids = [str(i) for i in train_ids]
    train_df = df.loc[train_ids, :]
    first_layer_components = cfg.first_layer_components
    second_layer_components = cfg.second_layer_components
    pca = PCA(
        n_components=first_layer_components + second_layer_components,
        svd_solver="randomized",
    )
    pca.fit(train_df.values)
    import matplotlib.ticker as ticker

    log.info(
        f"cumulative contribution rate: {np.cumsum(pca.explained_variance_ratio_)}"
    )
    plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    plt.plot([0] + list(np.cumsum(pca.explained_variance_ratio_)), "-o")
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative contribution rate")
    plt.grid()
    plt.savefig("pca.png")
    study_dir = Path(cfg.exp_dir) / "studies" / cfg.stage
    c_A = load_pca_weight(study_dir, cfg.morphing_start_case_id)
    c_B = load_pca_weight(study_dir, cfg.morphing_end_case_id)

    c_A_1, c_A_2 = c_A[first_layer_components:],c_A[:first_layer_components]
    c_B_1, c_B_2 = c_B[first_layer_components:],c_B[:first_layer_components]
    train_mean_mesh = train_df.mean(axis=0).values
    # まず最初にlayer1とlayer2同時のモーフィングを行う。
    if cfg.morphing_mode == "separate":

        for i in tqdm(range(cfg.interpolate_num + 1)):
            c_1 = c_A_1 * (1.0 - i / cfg.interpolate_num) + c_B_1 * (
                i / cfg.interpolate_num
            )
            c_2 = c_A_2 * (1.0 - i / cfg.interpolate_num) + c_B_2 * (
                i / cfg.interpolate_num
            )
            decoded_nodes = pca_reconstruct(np.concatenate([c_1, c_2]), pca, train_mean_mesh)
            tmp = OrganMesh(Path(cfg.template_mesh_path))
            tmp.nodes = decoded_nodes.reshape((-1, 3))
            tmp.save(f"layer_1_and_layer_2_{i}.ply")
        # 次にlayer1をhalfにして、layer2を変化させた時のモーフィングを行う。
        c_1 = (c_B_1 + c_A_1) / 2
        for i in tqdm(range(cfg.interpolate_num + 1)):
            c_2 = c_A_2 * (1.0 - i / cfg.interpolate_num) + c_B_2 * (
                i / cfg.interpolate_num
            )
            decoded_nodes = pca_reconstruct(np.concatenate([c_1, c_2]), pca, train_mean_mesh)
            tmp = OrganMesh(Path(cfg.template_mesh_path))
            tmp.nodes = decoded_nodes.reshape((-1, 3))
            tmp.save(f"layer_1_half_layer2_{i}.ply")
        # 最後にlayer2をhalfにして、layer1を変化させた時のモーフィングを行う。
        c_2 = (c_B_2 + c_A_2) / 2
        for i in tqdm(range(cfg.interpolate_num + 1)):
            c_1 = c_A_1 * (1.0 - i / cfg.interpolate_num) + c_B_1 * (
                i / cfg.interpolate_num
            )
            decoded_nodes = pca_reconstruct(np.concatenate([c_1, c_2]), pca, train_mean_mesh)
            tmp = OrganMesh(Path(cfg.template_mesh_path))
            tmp.nodes = decoded_nodes.reshape((-1, 3))
            tmp.save(f"layer_2_half_layer1_{i}.ply")
    elif cfg.morphing_mode == "grid":
        for i in tqdm(range(cfg.interpolate_num + 1)):
            for j in tqdm(range(cfg.interpolate_num + 1)):
                c_1 = c_A_1 * (1.0 - j / cfg.interpolate_num) + c_B_1 * (
                    j / cfg.interpolate_num
                ) # 高解像度
                c_2 = c_A_2 * (1.0 - i / cfg.interpolate_num) + c_B_2 * (
                    i / cfg.interpolate_num
                ) # 低解像度
                decoded_nodes = pca_reconstruct(np.concatenate([c_1, c_2]), pca, train_mean_mesh)
                tmp = OrganMesh(Path(cfg.template_mesh_path))
                tmp.nodes = decoded_nodes.reshape((-1, 3))
                tmp.save(f"{i}_{j}.ply")


def hmvae_morphing(cfg: DictConfig):
    template_mesh = OrganMesh(cfg.template_mesh_path)
    organ_name = OrganName(cfg.target_organ)
    latent_logger = latent_logger_factory(
        model_name=ModelName(cfg.model_name),
        experiment_id=cfg.experiment_id,
        number_of_vertices=template_mesh.num_nodes,
    )
    checkpoint_path = find_ckpt(Path(cfg.exp_dir))
    model = model_factory(
        model_name=ModelName(cfg.model_name),
        cfg=cfg,
        commit_hash=cfg.commit_hash,
        normal_logger=log,
        latent_logger=latent_logger,
        organ_name=organ_name,
        ckpt_path=checkpoint_path
    )
    model.model.to("cpu")
    z_A_1, z_A_2 = load_zs(Path(cfg.latent_dir), cfg.morphing_start_case_id, organ_name=organ_name)
    z_B_1, z_B_2 = load_zs(Path(cfg.latent_dir), cfg.morphing_end_case_id, organ_name=organ_name)
    tmp = OrganMesh(Path(cfg.subdivided_mesh2_path))
    if cfg.morphing_mode == "separate":
        # まず最初にlayer1とlayer2同時のモーフィングを行う。
        for i in tqdm(range(cfg.interpolate_num + 1)):
            z_1 = z_A_1 * (1.0 - i / cfg.interpolate_num) + z_B_1 * (
                i / cfg.interpolate_num
            )
            z_2 = z_A_2 * (1.0 - i / cfg.interpolate_num) + z_B_2 * (
                i / cfg.interpolate_num
            )
            model.eval()
            with torch.no_grad():
                decoded_nodes = model.model.decode(z_1, z_2)
            tmp = OrganMesh(Path(cfg.template_mesh_path))
            tmp.nodes = (
                decoded_nodes.to("cpu").detach().numpy().copy().reshape((-1, 3))
                * POSITIONAL_NORMALIZE_RATIO
            )
            tmp.save(f"layer_1_and_layer_2_{i}.ply")
        # 次にlayer1をhalfにして、layer2を変化させた時のモーフィングを行う。
        z_1 = (z_B_1 + z_A_1) / 2
        for i in tqdm(range(cfg.interpolate_num + 1)):
            z_2 = z_A_2 * (1.0 - i / cfg.interpolate_num) + z_B_2 * (
                i / cfg.interpolate_num
            )
            decoded_nodes = model.model.decode(z_1, z_2)
            tmp = OrganMesh(Path(cfg.template_mesh_path))
            tmp.nodes = (
                decoded_nodes.to("cpu").detach().numpy().copy().reshape((-1, 3))
                * POSITIONAL_NORMALIZE_RATIO
            )
            tmp.save(f"layer_1_half_layer2_{i}.ply")
        # 最後にlayer2をhalfにして、layer1を変化させた時のモーフィングを行う。
        z_2 = (z_B_2 + z_A_2) / 2
        for i in tqdm(range(cfg.interpolate_num + 1)):
            z_1 = z_A_1 * (1.0 - i / cfg.interpolate_num) + z_B_1 * (
                i / cfg.interpolate_num
            )
            decoded_nodes = model.model.decode(z_1, z_2)
            tmp = OrganMesh(Path(cfg.template_mesh_path))
            tmp.nodes = (
                decoded_nodes.to("cpu").detach().numpy().copy().reshape((-1, 3))
                * POSITIONAL_NORMALIZE_RATIO
            )
            tmp.save(f"layer_2_half_layer1_{i}.ply")
    elif cfg.morphing_mode == "grid":
        for i in tqdm(range(cfg.interpolate_num + 1)):
            for j in tqdm(range(cfg.interpolate_num + 1)):       
                z_1 = z_A_1 * (1.0 - j / cfg.interpolate_num) + z_B_1 * (
                    j / cfg.interpolate_num
                )
                z_2 = z_A_2 * (1.0 - i / cfg.interpolate_num) + z_B_2 * (
                    i / cfg.interpolate_num
                )
                # z_1 = torch.vstack([z_1, z_1, z_1])
                # z_2 = torch.vstack([z_2, z_2, z_2])
                decoded_nodes = model.model.decode(z_1, z_2)
                tmp = OrganMesh(Path(cfg.template_mesh_path))
                num_nodes = tmp.num_nodes
                tmp.nodes = (
                    decoded_nodes.to("cpu").detach().numpy().copy().reshape((num_nodes ,3))
                    * POSITIONAL_NORMALIZE_RATIO
                )
                tmp.save(f"{i}_{j}.ply")
    else:
        raise ValueError


if __name__ == "__main__":
    print("process start")
    main()
