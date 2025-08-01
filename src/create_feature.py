import logging
import os
from pathlib import Path

from typing import Optional, Literal
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from energy.energy import Energy
from energy.solver import NewtonEnergyMinimizer
from feature.DGandPositionFeature import DeformationGradients2DGandPositionFeature
from feature.discrete_laplacian import cal_discrete_laplacian
from mesh.mesh import OrganMesh
from mesh.edge_index import cal_edge_index
from common.entity import FeatureName, CaseID
from common.consts import (    POSITIONAL_NORMALIZE_RATIO,
    DISCRETE_LAPLACIAN_NORMALIZE_RATIO,
)
import numpy as np
import joblib

# A logger for this file
log = logging.getLogger(__name__)


def create_DGandPositionFeature(
    template_mesh: OrganMesh,
    target_mesh: OrganMesh,
    case_id: CaseID,
) -> Optional[np.ndarray]:
    try:
        ene = Energy(template_mesh=template_mesh, target_mesh=target_mesh)
    except ValueError:
        log.exception(f"case_id: {case_id.value} throw an error.")
        return None
    solver = NewtonEnergyMinimizer(energy=ene)
    # deformation gradients
    ts: list[np.ndarray] = []
    for i in tqdm(range(template_mesh.num_nodes)):
        optimize_result = solver.minimize(i)
        ts.append(optimize_result.x.reshape((3, 3)))

    feature = DeformationGradients2DGandPositionFeature().transform(
        ts, target_mesh=target_mesh
    )
    return feature


def create_DLandPositionFeature(
    template_mesh: OrganMesh,
    target_mesh: OrganMesh,
    normalization: Literal["sym", "rw", None],
) -> np.ndarray:
    edge_index = cal_edge_index(template_mesh=template_mesh)
    discrete_laplacian = (
        cal_discrete_laplacian(
            target_mesh, edge_index=edge_index, normalization=normalization
        )
        .to("cpu")
        .detach()
        .numpy()
        .copy()
    )
    feature_rows = []
    for i in tqdm(range(template_mesh.num_nodes)):
        positions = target_mesh.nodes[i].copy() / POSITIONAL_NORMALIZE_RATIO
        discrete_laplacian_feature = (
            discrete_laplacian[i].copy() / DISCRETE_LAPLACIAN_NORMALIZE_RATIO
        )
        feature_rows.append(
            np.concatenate([positions, discrete_laplacian_feature], axis=0)
        )
    return np.concatenate(feature_rows).reshape((-1, 6))


def create_DLandDGandPositionFeature(
    template_mesh: OrganMesh,
    target_mesh: OrganMesh,
    case_id: CaseID,
    normalization: Literal["sym", "rw", None],
) -> np.ndarray:
    """_summary_

    Args:
        template_mesh (OrganMesh): _description_
        target_mesh (OrganMesh): _description_
        case_id (CaseID): _description_
        normalization (Literal[&quot;sym&quot;, &quot;rw&quot;, None]): _description_

    Returns:
        np.ndarray: [num_vertices x (3+9+3)] position; deformation_gradient; discrete_laplacan
    """
    edge_index = cal_edge_index(template_mesh=template_mesh)
    discrete_laplacian = (
        cal_discrete_laplacian(
            target_mesh, edge_index=edge_index, normalization=normalization
        )
        .to("cpu")
        .detach()
        .numpy()
        .copy()
    )
    try:
        ene = Energy(template_mesh=template_mesh, target_mesh=target_mesh)
    except ValueError:
        log.exception(f"case_id: {case_id.value} throw an error.")
        return None
    solver = NewtonEnergyMinimizer(energy=ene)
    # deformation gradients
    ts: list[np.ndarray] = []
    for i in tqdm(range(template_mesh.num_nodes)):
        optimize_result = solver.minimize(i)
        ts.append(optimize_result.x.reshape((3, 3)))

    feature = (
        DeformationGradients2DGandPositionFeature()
        .transform(ts, target_mesh=target_mesh)
        .feature.reshape((-1, 12))
    )
    feature_rows = []
    for i in tqdm(range(template_mesh.num_nodes)):
        discrete_laplacian_feature = (
            discrete_laplacian[i].copy() / DISCRETE_LAPLACIAN_NORMALIZE_RATIO
        )
        feature_rows.append(
            np.concatenate([feature[i], discrete_laplacian_feature], axis=0)
        )
    return np.concatenate(feature_rows).reshape((-1, 15))


def create_feature(
    original_mesh_path: Path,
    created_feature_path: Path,
    template_mesh_path: Path,
    feature_name: FeatureName,
    case_id: str,
):
    if not original_mesh_path.exists():
        return
    target_mesh = OrganMesh(original_mesh_path)
    template_mesh = OrganMesh(template_mesh_path)
    if feature_name.value == FeatureName.DGandPosition.value:
        feature = create_DGandPositionFeature(
            template_mesh=template_mesh,
            target_mesh=target_mesh,
            case_id=case_id,
        )
        if feature is None:
            return None
        feature.save(created_feature_path)
    elif feature_name.value == FeatureName.DLandPosition.value:
        feature = create_DLandPositionFeature(
            template_mesh=template_mesh,
            target_mesh=target_mesh,
            normalization=None,
        )
        np.save(created_feature_path, feature)
    elif feature_name.value == FeatureName.DLandPositionRWNorm.value:
        feature = create_DLandPositionFeature(
            template_mesh=template_mesh,
            target_mesh=target_mesh,
            normalization="rw",
        )
        np.save(created_feature_path, feature)
    elif feature_name.value == FeatureName.DLandPositionSymNorm.value:
        feature = create_DLandPositionFeature(
            template_mesh=template_mesh,
            target_mesh=target_mesh,
            normalization="sym",
        )
        np.save(created_feature_path, feature)
    elif feature_name.value == FeatureName.DLandDGandPosition.value:
        feature = create_DLandDGandPositionFeature(
            template_mesh=template_mesh,
            target_mesh=target_mesh,
            case_id=case_id,
            normalization=None,
        )
        np.save(created_feature_path, feature)
    elif feature_name.value == FeatureName.DLRWNormandDGandPosition.value:
        feature = create_DLandDGandPositionFeature(
            template_mesh=template_mesh,
            target_mesh=target_mesh,
            case_id=case_id,
            normalization="rw",
        )
        np.save(created_feature_path, feature)
    else:
        raise ValueError(
            f"invalid feature name. You have to specify one of {list(FeatureName)} but you specified {feature_name}"
        )


@hydra.main(
    version_base="1.1",
    config_path="/work/src/conf",
    config_name="create_feature",
)
def main(cfg: DictConfig) -> None:
    data_dir = Path(cfg.data_dir)
    data_num = 0

    def wrap_function(case_id: CaseID):
        return create_feature(
            original_mesh_path=data_dir / case_id / f"00/output/{cfg.target_organ}.ply",
            created_feature_path=Path(
                f"{case_id}_{cfg.target_organ}_{cfg.feature_name}.npy"
            ),
            template_mesh_path=Path(cfg.template_path),
            feature_name=FeatureName(cfg.feature_name),
            case_id=case_id,
        )

    case_ids: list[CaseID] = []
    for case_id in os.listdir(data_dir):
        try:
            case_id = CaseID(case_id)
            case_ids.append(case_id)
        except AssertionError:
            continue
    if cfg.debug_mode:
        case_ids = case_ids[:5]

    if cfg.parallel:

        joblib.Parallel(n_jobs=-1)(
            joblib.delayed(wrap_function)(case_id.value) for case_id in case_ids
        )
    else:
        for case_id in case_ids:
            wrap_function(case_id=case_id.value)

    log.info(
        f"{data_num} {cfg.target_organ} .ply files are converted into {cfg.feature_name} features."
    )


if __name__ == "__main__":
    main()
