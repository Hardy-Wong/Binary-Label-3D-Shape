from enum import Enum
from typing import Tuple

import numpy as np
from registration import GlobalRotationRegisration, PointToPointICP

from src.mesh.mesh import OrganMesh


class EvaluateReconstractionMethod(Enum):
    p2p = "p2p"
    global_rotation = "global_rotation"


def evaluate_reconstruction(
    source: OrganMesh, target: OrganMesh, method: EvaluateReconstractionMethod
) -> Tuple[float, OrganMesh]:
    if method == EvaluateReconstractionMethod.global_rotation:
        registration = GlobalRotationRegisration(source=source, target=target)
    elif method == EvaluateReconstractionMethod.p2p:
        registration = PointToPointICP(source=source, target=target)
    else:
        raise NotImplementedError
    registration_source = registration.transform()
    err = 0.0
    for i in range(source.num_nodes):
        err += np.linalg.norm(registration_source.nodes[i] - target.nodes[i])

    return err / source.num_nodes, registration_source
