from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import numpy as np
from tqdm import tqdm

from src.mesh.mesh import OrganMesh

if TYPE_CHECKING:
    from src.feature.RIMDfeature import RIMDFeatureInterface


class Energy:
    def __init__(self, template_mesh: OrganMesh, target_mesh: OrganMesh):
        if template_mesh.num_nodes != target_mesh.num_nodes:
            raise ValueError(
                f"template_mesh nodes {template_mesh.num_nodes} != target_mesh nodes {target_mesh.num_nodes}"
            )
        self.template_mesh = template_mesh
        self.target_mesh = target_mesh

    def call(self, i: int, deformation_gradient: np.ndarray) -> float:
        """
        最小化すべきエネルギーを計算。
        """
        energy = 0.0
        adjacent_vs = self.target_mesh.get_vertex_adjacent_vertices(i)
        for j in adjacent_vs:
            e_dash = self.target_mesh.nodes[i] - self.target_mesh.nodes[j]
            e = self.template_mesh.nodes[i] - self.template_mesh.nodes[j]
            # print(e_dash - e)
            square_err = (
                (e_dash - deformation_gradient.reshape((3, 3)) @ e) ** 2
            ).sum()
            cot_weight = self.template_mesh.get_coweight(i=i, j=j)
            energy += cot_weight * square_err
        energy /= len(adjacent_vs)
        return energy

    def all_call(
        self, feature_interface: RIMDFeatureInterface, Rs: List[Optional[np.ndarray]]
    ) -> float:
        dst = 0.0
        for i in tqdm(range(self.template_mesh.num_nodes)):
            for j in self.template_mesh.get_vertex_adjacent_vertices(i):
                cij = self.template_mesh.get_coweight(i, j)
                e_dash = self.target_mesh.nodes[i] - self.target_mesh.nodes[j]
                e = self.template_mesh.nodes[i] - self.template_mesh.nodes[j]
                tmp = 0.0
                for t in self.target_mesh.get_vertex_adjacent_vertices(i):
                    Tt = Rs[t] @ feature_interface.dRij(t, i) @ feature_interface.Si(i)
                    square_err = ((e_dash - Tt.reshape((3, 3)) @ e) ** 2).sum()
                    tmp += square_err / len(
                        self.target_mesh.get_vertex_adjacent_vertices(i)
                    )
                dst += tmp * cij
        return dst
