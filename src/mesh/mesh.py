from functools import lru_cache
from pathlib import Path
from typing import List

import numpy as np
import open3d as o3d
import trimesh


class OrganMesh:
    def __init__(self, mesh_path: Path) -> None:
        self.mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        self.update_from_o3d_mesh()
        self.mesh_path = mesh_path

    def save(self, file_name: str) -> None:
        mesh = trimesh.Trimesh(vertices=self.nodes, faces=self.faces)
        mesh.export(file_name)

    def update_from_o3d_mesh(self) -> None:
        """self.meshが更新された際に、残りのプロパティを更新するメソッド"""
        self.nodes = np.asarray(self.mesh.vertices)
        self.faces = np.asarray(self.mesh.triangles)
        self.mesh.compute_adjacency_list()
        self.num_nodes: int = len(self.nodes)
        self.mesh_path = None

    @lru_cache(maxsize=10**5)
    def get_vertex_adjacent_vertices(self, i) -> List[int]:
        """
        隣接するnodeのidのListを返却
        """
        return sorted(self.mesh.adjacency_list[i])

    @lru_cache(maxsize=10000)
    def get_coweight(self, i: int, j: int):
        adjacency_list = self.mesh.adjacency_list
        i_neighbor, j_neighbor = adjacency_list[i], adjacency_list[j]
        rest = list(i_neighbor & j_neighbor)
        assert len(rest) == 2

        cot_alpha = self.__cal_cot(
            (self.nodes[i] - self.nodes[rest[0]]), (self.nodes[j] - self.nodes[rest[0]])
        )
        cot_beta = self.__cal_cot(
            (self.nodes[i] - self.nodes[rest[1]]), (self.nodes[j] - self.nodes[rest[1]])
        )

        return abs(cot_alpha + cot_beta)

    def __cal_cot(self, vi: np.ndarray, vj: np.ndarray):
        cos = np.inner(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj))
        theta = np.arccos(cos)
        return 1.0 / np.tan(theta)
