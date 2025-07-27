import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from src.mesh.mesh import OrganMesh


def draw_organ(fig: Figure, ax: plt.Axes, organ_mesh: OrganMesh, **kwrgs) -> Line2D:
    x, y, z = organ_mesh.nodes[:, 0], organ_mesh.nodes[:, 1], organ_mesh.nodes[:, 2]
    return ax.scatter(x, y, z, **kwrgs)
