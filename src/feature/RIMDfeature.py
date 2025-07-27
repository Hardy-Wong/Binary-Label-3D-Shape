from pathlib import Path
from typing import List, Optional

import numpy as np
from scipy.linalg import expm, polar, svd
from sklearn.metrics import mean_squared_error

from src.common.entity import CaseID, OrganName
from src.energy.energy import Energy
from src.mesh.mesh import OrganMesh


class RIMDFeature:
    """https://dl.acm.org/doi/10.1145/2908736"""

    def __init__(self, feature: np.ndarray) -> None:
        self.feature = feature

    def save(self, file: Path):
        if self.feature is None:
            raise ValueError("feature content is Empty!")
        np.save(file=file, arr=self.feature)

    @classmethod
    def load(cls, file: Path):
        feature: np.ndarray = np.load(file)
        return cls(feature=feature)


class DeformationGradients2RIMDFeature:
    def __init__(self, epsilon=1e-5) -> None:
        self.epsilon = epsilon

    def transform(
        self, deformation_gradients: List[np.ndarray], template_mesh: OrganMesh
    ) -> RIMDFeature:
        raw_features = []
        rotation_matrixes = []
        stretch_matrixes = []
        for t in deformation_gradients:
            r, s = polar(t)
            rotation_matrixes.append(r)
            stretch_matrixes.append(s)

        for i in range(template_mesh.num_nodes):
            adjacent_node_idx = template_mesh.get_vertex_adjacent_vertices(i)
            Ri = rotation_matrixes[i]
            Si = stretch_matrixes[i]
            r_features = []
            for j in adjacent_node_idx:
                Rj = rotation_matrixes[j]
                log_dRij = self.cal_logR(Ri.T @ Rj)
                log_R_triu = np.triu(log_dRij)
                r_feature = np.concatenate([log_R_triu[0][1:3], [log_R_triu[1][2]]])
                # log_Rの上三角部分の3つの要素を抜き出している。
                r_features.extend(r_feature)

            s_features = np.concatenate([Si[0][:3], Si[1][1:], [Si[2][-1]]])
            raw_features.append(np.concatenate([s_features, np.array(r_features)]))
        raw_features = np.array(raw_features)
        return RIMDFeature(feature=np.concatenate(raw_features, axis=0))

    def cal_logR(self, R: np.ndarray) -> np.ndarray:
        """calculate logarithm of rotation matrix

        Args:
            R (np.ndarray): rotation matrix (3x3)

        Returns:
            np.ndarray: logR
        """
        theta = np.arccos((np.trace(R) - 1) / 2)
        if np.sin(theta) < self.epsilon:
            return np.zeros((3, 3))
        # zero - divisionを防ぐ
        return (theta / (2 * (np.sin(theta)))) * (R - R.T)


class RIMDFeatureInterface:
    def __init__(self, template_mesh: OrganMesh, feature: RIMDFeature) -> None:
        self.template_mesh = template_mesh
        self.feature = feature

        self.vertex_feature_idx = []
        curr_idx = 0
        for i in range(template_mesh.num_nodes):
            self.vertex_feature_idx.append(curr_idx)
            curr_idx += 6 + len(self.template_mesh.get_vertex_adjacent_vertices(i)) * 3
            # Siは6成分、dRijは３成分だから

    def Si(self, i) -> np.ndarray:
        f_idx = self.vertex_feature_idx[i]
        f = self.feature.feature[f_idx : f_idx + 6]
        dst = np.zeros((3, 3))
        dst[0] = f[:3]
        dst[1][1] = f[3]
        dst[1][0] = f[4]
        dst[1][2] = f[4]
        dst[2][0] = f[2]
        dst[2][1] = f[4]
        dst[2][2] = f[5]
        return dst

    def log_dRij(self, i, j) -> np.ndarray:
        idx_in_R = self.template_mesh.get_vertex_adjacent_vertices(i).index(j)
        f_idx = self.vertex_feature_idx[i] + 6 + 3 * idx_in_R  # Siで6成分。Rは3成分のみ
        f = self.feature.feature[f_idx : f_idx + 3]
        dst = np.zeros((3, 3))
        dst[0][1] = f[0]
        dst[0][2] = f[1]
        dst[1][2] = f[2]
        return dst - dst.T

    def dRij(self, i, j) -> np.ndarray:
        log_dRij = self.log_dRij(i, j)
        return expm(log_dRij)


class RIMDFeature2Mesh:
    def __init__(
        self,
        feature_interface: RIMDFeatureInterface,
        organ: OrganName = OrganName.liver,
        case_id: CaseID = CaseID(1),
    ) -> None:
        self.template_mesh = feature_interface.template_mesh
        self.feature = feature_interface.feature
        self.feature_interface = feature_interface
        self.organ = organ
        self.case_id = case_id
        self.A = self.__construct_A_in_global_step()
        self.A_inverse = np.linalg.pinv(self.A)
        self.__bfs_initialize()

    def __bfs_initialize(self):
        """rotation_matrixを初期化する関数"""
        visited = set()
        initialized_rotation_matrix: List[Optional[np.ndarray]] = [
            None for i in range(self.template_mesh.num_nodes)
        ]
        initialized_rotation_matrix[0] = np.identity(3)
        visited.add(0)
        while len(visited) < self.template_mesh.num_nodes:
            i = visited.pop()
            adjacency_nodes = self.template_mesh.get_vertex_adjacent_vertices(i)
            for j in adjacency_nodes:
                dRij = self.feature_interface.dRij(i, j)
                Rj = initialized_rotation_matrix[i] @ dRij  # Rj = Ri @ dRij
                initialized_rotation_matrix[j] = Rj
                visited.add(j)

            visited.add(i)

        self.Rs = initialized_rotation_matrix

    def transform(
        self, target_mesh: OrganMesh, threshold: float = 1e-5, logger=None
    ) -> OrganMesh:
        """RIMD特徴量からメッシュへの変換を行うスクリプト。

        Args:
            target_mesh (OrganMesh): this mesh will be changed inplace.
            threshold ( float ): threshold for determine whether optimization has been finished or not.
        Returns:
            None : this method will change the input target_mesh inplace.
        """
        init_ene = 0
        diff = float("inf")
        i = 0
        while diff > threshold:
            i += 1
            self.global_step(target_mesh=target_mesh)
            self.local_step(target_mesh=target_mesh)
            ene = Energy(
                template_mesh=self.template_mesh, target_mesh=target_mesh
            ).all_call(feature_interface=self.feature_interface, Rs=self.Rs)
            diff = abs(init_ene - ene)
            print(ene)
            init_ene = ene
            if i > 100 and logger:
                logger.warn(
                    f"{self.organ.value} case_id: {self.case_id.value} shape reconstruction is not converged. Energy diff is {diff}"
                )
                return target_mesh
        logger.info(
            f"{self.organ.value} case_id: {self.case_id.value} shape reconstruction is converged. number of iteration is {i}"
        )
        return target_mesh

    def global_step(self, target_mesh: OrganMesh) -> None:
        b = self.construct_b_in_global_step()
        target_mesh.nodes = self.A_inverse @ b / 2.0

    def __construct_A_in_global_step(self) -> np.ndarray:
        A = np.identity(self.template_mesh.num_nodes)
        for j in range(self.template_mesh.num_nodes):
            Nj = self.template_mesh.get_vertex_adjacent_vertices(j)
            ajj = 0.0
            for k in Nj:
                cjk = self.template_mesh.get_coweight(j, k)
                A[j][k] = -cjk
                ajj += cjk
            A[j][j] = ajj
        return A

    def construct_b_in_global_step(self) -> np.ndarray:
        b_rows = []
        for j in range(self.template_mesh.num_nodes):
            b_row = np.zeros(3)
            for k in self.template_mesh.get_vertex_adjacent_vertices(j):
                ejk = self.template_mesh.nodes[j] - self.template_mesh.nodes[k]
                cjk = self.template_mesh.get_coweight(j, k)
                pre, later = np.zeros((3, 3)), np.zeros((3, 3))
                for s in self.template_mesh.get_vertex_adjacent_vertices(k):
                    pre += self.__Rs_dRsk_Sk(s=s, k=k)
                for i in self.template_mesh.get_vertex_adjacent_vertices(j):
                    later += self.__Rs_dRsk_Sk(s=i, k=j)

                pre, later = pre / len(
                    self.template_mesh.get_vertex_adjacent_vertices(k)
                ), later / len(self.template_mesh.get_vertex_adjacent_vertices(j))

                tmp = cjk * (pre + later) @ ejk
                b_row += tmp

            b_rows.append(b_row)

        return np.vstack(b_rows)

    def __Rs_dRsk_Sk(self, s: int, k: int) -> np.ndarray:
        Rs = self.Rs[s]
        dRsk = self.feature_interface.dRij(s, k)
        Sk = self.feature_interface.Si(k)
        return Rs @ dRsk @ Sk

    def local_step(self, target_mesh: OrganMesh) -> None:
        for i in range(target_mesh.num_nodes):
            Qi = np.zeros((3, 3))
            for j in target_mesh.get_vertex_adjacent_vertices(i):
                dRij = self.feature_interface.dRij(i, j)
                Sj = self.feature_interface.Si(j)
                tmp = np.zeros((3, 3))
                for k in self.template_mesh.get_vertex_adjacent_vertices(j):
                    tmp += self.__cjk_ejk_ejk_dash(target_mesh=target_mesh, j=j, k=k)
                Qi += (dRij @ Sj @ tmp) / len(
                    target_mesh.get_vertex_adjacent_vertices(j)
                )

            Ui, _, Vi = svd(Qi)
            Ri = Vi.T @ Ui.T
            if np.linalg.det(Ri) < 0:
                Ri = -Ri
            self.Rs[i] = Ri.copy()
            if i == 0:
                # TODO: テストが終わったら消す
                Q0 = Qi
        return Q0

    def __cjk_ejk_ejk_dash(self, target_mesh: OrganMesh, j: int, k: int):
        cjk = self.template_mesh.get_coweight(j, k)
        ejk = self.template_mesh.nodes[j] - self.template_mesh.nodes[k]
        ejk_dash = target_mesh.nodes[j] - target_mesh.nodes[k]
        return cjk * np.array([ejk]).T @ np.array([ejk_dash])

    def __recon_error(
        self, before_mesh_nodes: np.ndarray, current_mesh_nodes: np.ndarray
    ) -> float:
        return mean_squared_error(before_mesh_nodes, current_mesh_nodes)
