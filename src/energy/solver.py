import numpy as np
from scipy.optimize import OptimizeResult, minimize

from src.energy.energy import Energy


class NewtonEnergyMinimizer:
    def __init__(self, energy: Energy) -> None:
        self.energy = energy

    def minimize(self, i: int) -> OptimizeResult:
        init_deformation_gradient = np.identity(3)
        res = minimize(
            lambda t: self.energy.call(i, t), init_deformation_gradient, method="Powell"
        )
        return res
