from pathlib import Path
import numpy as np
import hydra
from omegaconf import DictConfig
from visualize.color import VisualizeColorRGB
import trimesh


depth_colors = [
    VisualizeColorRGB(255, 255, 255),
    VisualizeColorRGB(66, 245, 230),
    VisualizeColorRGB(66, 147, 245),
    VisualizeColorRGB(0, 0, 0),
]


def color_to_rgb_arr(color: VisualizeColorRGB) -> np.ndarray:
    return np.array(
        [
            color.r,
            color.g,
            color.b,
        ]
    )


@hydra.main(version_base="1.1", config_path="conf", config_name="subdivide")
def main(cfg: DictConfig):
    base_mesh_path = Path(cfg.base_mesh_path)
    depth = cfg.depth
    output_filename_prefix = cfg.output_filename_prefix
    base_mesh = trimesh.load(str(base_mesh_path))
    v_depth_idxes = [len(base_mesh.vertices)]
    base_mesh.export(f"{output_filename_prefix}_0.ply")
    for d in range(1, depth):
        base_mesh: trimesh.Trimesh = base_mesh.subdivide()
        vertexes_colors = []
        for i in range(d):
            vertex_idxes = v_depth_idxes[i]
            color = depth_colors[i]
            vertexes_color = [color_to_rgb_arr(color)] * vertex_idxes
            vertexes_colors.extend(vertexes_color)
        v_depth_idxes.append(len(base_mesh.vertices) - v_depth_idxes[-1])
        base_mesh.visual.vertex_colors = vertexes_colors
        base_mesh.export(f"{output_filename_prefix}_{d}.ply")


if __name__ == "__main__":
    main()
