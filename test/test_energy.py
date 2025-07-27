import sys

sys.path.append("/work/src")
from pathlib import Path

import numpy as np

from src.energy.energy import Energy
from src.mesh.mesh import OrganMesh

mesh = OrganMesh(Path(__file__).parent / "test_tetra.ply")
target = OrganMesh(Path(__file__).parent / "test_tetra.ply")


def test_call():
    energy = Energy(template_mesh=mesh, target_mesh=target)
    deformation_gradients = [np.identity(3).reshape(9) for i in range(mesh.num_nodes)]
    for i, deformation_gradient in enumerate(deformation_gradients):
        assert 0.0 == energy.call(i, deformation_gradient=deformation_gradient)
