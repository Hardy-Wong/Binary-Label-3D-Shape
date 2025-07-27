import logging
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from src.energy.energy import Energy
from src.energy.solver import NewtonEnergyMinimizer
from src.feature.DGandPositionFeature import DeformationGradients2DGandPositionFeature
from src.mesh.mesh import OrganMesh

# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(
    version_base="1.1",
    config_path="/work/src/conf",
    config_name="create_DGandPositionFeature",
)
def main(cfg: DictConfig) -> None:
    if cfg.commit_id == "none-select":
        raise ValueError(
            "commit id should be specified! please run GIT_COMMIT_HASH=$(git rev-parse HEAD)"
        )

    template_mesh = OrganMesh(Path(cfg.template_path))
    data_dir = Path(cfg.data_dir)
    data_num = 0
    for case_id in os.listdir(data_dir):
        target_mesh_path = data_dir / case_id / f"00/output/{cfg.target_organ}.ply"
        if not target_mesh_path.exists():
            continue
        target_mesh = OrganMesh(target_mesh_path)
        try:
            ene = Energy(template_mesh=template_mesh, target_mesh=target_mesh)
        except ValueError:
            log.exception(f"case_id: {case_id} throw an error.")
            continue
        solver = NewtonEnergyMinimizer(energy=ene)
        ts = []
        for i in tqdm(range(template_mesh.num_nodes)):
            optimize_result = solver.minimize(i)
            ts.append(optimize_result.x.reshape((3, 3)))
        feature = DeformationGradients2DGandPositionFeature().transform(
            ts, target_mesh=target_mesh
        )
        feature.save(Path(f"{case_id}_{cfg.target_organ}_DGandPosition.npy"))
        data_num += 1

        if cfg.debug_mode and data_num == 20:
            break

    log.info(
        f"{data_num} {cfg.target_organ} .ply files are converted into DGandPosition features."
    )


if __name__ == "__main__":
    main()
