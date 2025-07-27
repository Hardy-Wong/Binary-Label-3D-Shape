import torch

from common.consts import NORMALIZE_RATIO


class PositionalDataAugmentation:
    def __init__(self, ratio: float = 1 / NORMALIZE_RATIO) -> None:
        self.ratio = ratio

    def __call__(self, feature: torch.Tensor) -> torch.Tensor:
        feature_copy = feature.clone()
        rand_position = torch.rand([3]) * self.ratio
        for idx in range(len(feature_copy) // 12):
            feature_copy[idx * 12 : idx * 12 + 3] += rand_position
        return feature_copy
