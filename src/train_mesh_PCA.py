from os import listdir,makedirs
from pathlib import Path
import pickle

from common.consts import POSITIONAL_NORMALIZE_RATIO 
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import optuna

from common.entity import CaseID
from mesh.mesh import OrganMesh
from common.entity import OrganName
from common.splited_data import load_case_ids 
from post_process import post_process
import logging
from tqdm import tqdm
import ray

NUM_NODES = 738
DEBUG_MESH_NUM = 5

@ray.remote
def predict_1mesh(mesh: np.ndarray, pca: PCA, mean_mesh: np.ndarray, case_id: str) -> tuple[np.ndarray, optuna.Study]:
    """reconstruct mesh from pca model. 
    Args:
        mesh (OrganMesh): mesh to reconstruct
        pca (PCA): trained pca model
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    mesh = mesh.reshape((-1,3))
    mean_mesh = mean_mesh.reshape((-1,3))
    eigen_vectors = pca.components_
    # optimize weight vector with optuna in order to minimize reconstruction error between original mesh and reconstructed mesh
    def objective(trial) -> float:
        weight_vector = np.array([trial.suggest_uniform(f"weight_{i}", -1, 1) for i in range(len(eigen_vectors))])
        non_centered_result = (weight_vector @ eigen_vectors).reshape((NUM_NODES,3)) * POSITIONAL_NORMALIZE_RATIO
        return np.linalg.norm(mesh - (mean_mesh + non_centered_result))/NUM_NODES
    
    study = optuna.create_study()
    study.optimize(objective, n_trials=100, show_progress_bar=False)
    weight_vector = np.array([study.best_params[f"weight_{i}"] for i in range(len(eigen_vectors))])
    non_centered_result = (weight_vector @ eigen_vectors).reshape((NUM_NODES, 3)) * POSITIONAL_NORMALIZE_RATIO
    result = mean_mesh + non_centered_result
    return result.flatten(), study, case_id

def predict_meshes(meshes: dict[str,np.ndarray], pca: PCA, mean_mesh: np.ndarray) -> tuple[list[np.ndarray], dict[str,optuna.Study]]:
    """reconstruct meshes from pca model. 
    Args:
        meshes (dict[str,np.ndarray]): meshes to reconstruct
        pca (PCA): trained pca model
        mean_mesh: np.ndarray
    """
    ids: list[str] = []
    for case_id, mesh in tqdm(meshes.items()):
        ray_id = predict_1mesh.remote(mesh, pca, mean_mesh, str(case_id))
        ids.append(ray_id)
    res = ray.get(ids)
    results: list[np.ndarray] = []
    studies: dict[str, optuna.Study] = dict()
    for result, study, case_id in res:
        studies[case_id] = study
        results.append(result)
    return results, studies



log = logging.getLogger(__name__)



def load_organ_nodes_dataframe(
    organ_name: OrganName,
    data_dir=Path("/home/snail/mesh_gcn_vae/data/3D-CT PLY"),
    emit: bool = False,
) -> pd.DataFrame:
    vs = []
    case_ids = []
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
    config_path="conf",
    config_name="train_mesh_PCA.yaml",
)
def main(cfg: DictConfig):
    ray.init(num_cpus=10)
    df = load_organ_nodes_dataframe(
        organ_name=OrganName(cfg.target_organ),
        emit=True,
    )
    train_ids = load_case_ids(mode="train")
    val_ids = load_case_ids(mode="val")
    test_ids = load_case_ids(mode="test")
    if cfg.debug_mode:
        train_ids = train_ids[:DEBUG_MESH_NUM]
        val_ids = val_ids[:DEBUG_MESH_NUM]
        test_ids = test_ids[:DEBUG_MESH_NUM]
    train_ids = [str(i) for i in train_ids]
    train_df = df.loc[train_ids, :]
    val_ids = [str(i) for i in val_ids]
    val_df = df.loc[val_ids, :]
    test_ids = [str(i) for i in test_ids]
    test_df = df.loc[test_ids, :]
    pca = PCA(
        n_components= DEBUG_MESH_NUM if cfg.debug_mode else cfg.num_components,
        svd_solver="randomized",
    )
    pca.fit(train_df.values/POSITIONAL_NORMALIZE_RATIO)
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
    train_mean_mesh = train_df.mean(axis=0).values
    train_meshes = {CaseID(case_id): meshes for case_id, meshes in zip(train_df.index,train_df.values)}
    train_preds, train_studies = predict_meshes(train_meshes,pca,train_mean_mesh)
    train_pred_df = pd.DataFrame(
        data=train_preds,
        index=train_df.index,
        columns=train_df.columns,
    )
    train_preds = [
        pd.DataFrame(data={
            "estimated_x": es_row[0::3].values,
            "estimated_y": es_row[1::3].values,
            "estimated_z": es_row[2::3].values,
            "gt_x": gt_row[0::3].values,
            "gt_y": gt_row[1::3].values,
            "gt_z": gt_row[2::3].values,
            "diff_x": ( es_row[0::3] - gt_row[0::3] ).values,
            "diff_y": ( es_row[1::3] - gt_row[1::3] ).values,
            "diff_z": ( es_row[2::3] - gt_row[2::3] ).values,
            "case_id": case_id,
            "vertex_id": [i for i in range(len(es_row) // 3)],
            "epoch": 1000,
            "batch_idx": 0,
            "mode": "train"
        }) for (case_id, es_row), (_,gt_row) in zip(train_pred_df.iterrows(),train_df.iterrows())
    ]
    # merge into 1 dataframe
    
        
    val_meshes = {CaseID(case_id): meshes for case_id, meshes in zip(val_df.index,val_df.values)}
    val_preds, val_studies = predict_meshes(val_meshes,pca,train_mean_mesh)
    val_pred_df = pd.DataFrame(
        data=val_preds,
        index=val_df.index,
        columns=val_df.columns,
    )
    val_preds = [
        pd.DataFrame(data={
            "estimated_x": es_row[0::3].values,
            "estimated_y": es_row[1::3].values,
            "estimated_z": es_row[2::3].values,
            "gt_x": gt_row[0::3].values,
            "gt_y": gt_row[1::3].values,
            "gt_z": gt_row[2::3].values,
            "diff_x": ( es_row[0::3] - gt_row[0::3] ).values,
            "diff_y": ( es_row[1::3] - gt_row[1::3] ).values,
            "diff_z": ( es_row[2::3] - gt_row[2::3] ).values,
            "case_id": case_id,
            "vertex_id": [i for i in range(len(es_row) // 3)],
            "epoch": 1000,
            "batch_idx": 0,
            "mode": "val"
        }) for (case_id, es_row), (_,gt_row) in zip(val_pred_df.iterrows(),val_df.iterrows())
    ]
    
    test_meshes = {CaseID(case_id): meshes for case_id, meshes in zip(test_df.index,test_df.values)}
    test_preds, test_studies = predict_meshes(test_meshes,pca,train_mean_mesh)
    test_pred_df = pd.DataFrame(
        data=test_preds,
        index=test_df.index,
        columns=test_df.columns,
    )
    test_preds = [
        pd.DataFrame(data={
            "estimated_x": es_row[0::3].values,
            "estimated_y": es_row[1::3].values,
            "estimated_z": es_row[2::3].values,
            "gt_x": gt_row[0::3].values,
            "gt_y": gt_row[1::3].values,
            "gt_z": gt_row[2::3].values,
            "diff_x": ( es_row[0::3] - gt_row[0::3] ).values,
            "diff_y": ( es_row[1::3] - gt_row[1::3] ).values,
            "diff_z": ( es_row[2::3] - gt_row[2::3] ).values,
            "case_id": case_id,
            "vertex_id": [i for i in range(len(es_row) // 3)],
            "epoch": 1000,
            "batch_idx": 0,
            "mode": "test"
        }) for (case_id, es_row), (_,gt_row) in zip(test_pred_df.iterrows(),test_df.iterrows())
    ]
    ray.shutdown()
    studies_dir = Path("studies")
    makedirs(studies_dir,exist_ok=True)
    makedirs(studies_dir/"train",exist_ok=True)
    makedirs(studies_dir/"test",exist_ok=True)
    makedirs(studies_dir/"val",exist_ok=True)
    for case_id, study in train_studies.items():
        study_path = studies_dir/"train"/f"{case_id}.pkl"
        with open(study_path, "wb") as f:
            pickle.dump(study, f)
    for case_id, study in test_studies.items():
        study_path = studies_dir/"test"/f"{case_id}.pkl"
        with open(study_path, "wb") as f:
            pickle.dump(study, f)
    for case_id, study in val_studies.items():
        study_path = studies_dir/"val"/f"{case_id}.pkl"
        with open(study_path, "wb") as f:
            pickle.dump(study, f)

    taraget_dir = Path("log_vertex_position_diff_df")
    makedirs(taraget_dir,exist_ok=True)
    for train_pred in train_preds:
        train_pred.to_feather(taraget_dir/ f"_train_{train_pred['case_id'].values[0]}_1000ep_0batch.feather")
    for val_pred in val_preds:
        val_pred.to_feather(taraget_dir/ f"_val_{val_pred['case_id'].values[0]}_1000ep_0batch.feather")
    for test_pred in test_preds:
        test_pred.to_feather(taraget_dir/ f"_test_{test_pred['case_id'].values[0]}_1000ep_0batch.feather")
    post_process(
        log_vertex_position_df_dir=taraget_dir.resolve(
            strict=True
        ),
        commit_hash=cfg.experiment_id,
        template_mesh_path=Path(cfg.template_mesh_path),
        organ_name=OrganName(cfg.target_organ),
        will_plot_metrics=cfg.will_plot_metrics,
        will_plot_samples=cfg.will_plot_samples,
    )

if __name__ == "__main__":
    main()


