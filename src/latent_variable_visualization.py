from os import listdir, environ
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import hydra
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from omegaconf import DictConfig
import torch

from common.entity import ModelName, CaseID
from mesh.mesh import OrganMesh
from common.entity import OrganName
from common.consts import POSITIONAL_NORMALIZE_RATIO
from models.factories.model_factory import model_factory
from models.factories.latent_logger_factory import latent_logger_factory
from common.splited_data import load_case_ids
import logging
from tqdm import tqdm


log = logging.getLogger(__name__)


def load_zs(data_dir: Path, case_id: CaseID, organ_name: OrganName):
    file_name = f"_{case_id}_{organ_name}_latent_vec.npy"
    z1 = np.load(data_dir / "layer_1" / file_name)
    z2 = np.load(data_dir / "layer_2" / file_name)
    return z1, z2

def zs_pca(z1, z2):
    pca = PCA(n_components=1)
    pca.fit(z1)
    z1_main = pca.transform(z1)
    pca.fit(z2)
    z2_main = pca.transform(z2)
    return z1_main, z2_main

test_case_ids = ["006","046","047","049","050","051","055","065","068","075","080","085","087","090","094","099","127","132","144"]
test_Z1 = []
test_Z2 = []
for i in range(19):
    test_id = test_case_ids[i]
    print(test_id)
    Z1, Z2 = load_zs(Path("/home/snail/mesh_gcn_vae/artifact/stomach_latent_vecs/latent_vecs/test"), "case%s" %test_id, "stomach" )
    Z1 = Z1.reshape(-1,)
    Z2 = Z2.reshape(-1,)
    test_Z1.append(Z1)
    test_Z2.append(Z2)
test_Z1 = np.array(test_Z1)
test_Z2 = np.array(test_Z2)
x, y = zs_pca(test_Z1, test_Z2)
print(x)
print(y)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Latent Variable '+r'$z_1$', fontsize=28)
plt.ylabel('Latent Variable '+r'$z_2$', fontsize=28)
plt.scatter(x,y, s=120)
plt.tight_layout()
plt.show()
plt.savefig("/home/snail/mesh_gcn_vae/example.png")