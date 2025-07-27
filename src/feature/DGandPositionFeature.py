from pathlib import Path
from typing import List

import numpy as np
from scipy.linalg import expm, polar

from src.common.consts import NORMALIZE_RATIO
from src.mesh.mesh import OrganMesh


class DGandPositionFeature:
    """DG(DeformationGradient) and Position Feature"""

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


class DeformationGradients2DGandPositionFeature:
    def __init__(self, epsilon=1e-5, normalize_ratio: float = NORMALIZE_RATIO) -> None:
        self.epsilon = epsilon
        self.normalize_ratio = normalize_ratio

    def transform(
        self, deformation_gradients: List[np.ndarray], target_mesh: OrganMesh
    ) -> DGandPositionFeature:
        raw_features = []
        rotation_matrixes = []
        stretch_matrixes = []
        for t in deformation_gradients:
            r, s = polar(t)
            rotation_matrixes.append(r)
            stretch_matrixes.append(s)

        for i in range(target_mesh.num_nodes):
            Ri = rotation_matrixes[i]
            r_features = []
            adjacent_node_idx = target_mesh.get_vertex_adjacent_vertices(i)
            for j in adjacent_node_idx:
                Rj = rotation_matrixes[j]
                log_dRij = self.cal_logR(Ri.T @ Rj)
                log_R_triu = np.triu(log_dRij)
                r_features.append(self.Rotation_feature_extract(logR=log_R_triu))
            r_feature = np.average(r_features, axis=0)
            Si = stretch_matrixes[i]
            s_feature = self.Stretch_feature_extract(Si)
            raw_features.append(
                np.concatenate(
                    [target_mesh.nodes[i] / self.normalize_ratio, r_feature, s_feature]
                )
            )
        raw_features = np.array(raw_features)
        return DGandPositionFeature(feature=np.concatenate(raw_features, axis=0))

    def Rotation_feature_extract(self, logR: np.ndarray) -> np.ndarray:
        return np.array([logR[0][1], logR[0][2], logR[1][2]])

    def Stretch_feature_extract(self, S: np.ndarray) -> np.ndarray:
        return np.array([S[0][0], S[0][1], S[0][2], S[1][1], S[1][2], S[2][2]])

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


class DGandPositionFeatureInterface:
    def __init__(self, template_mesh: OrganMesh, feature: DGandPositionFeature) -> None:
        self.template_mesh = template_mesh
        self.feature = feature

    def Si(self, i) -> np.ndarray:
        f = self.feature.feature[i][6:]
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
