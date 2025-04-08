import nichepca as npc
import scanpy as sc
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch_geometric as pyg

from spatialnca.config import Config
from spatialnca.utils import construct_graph


def prepare_data(adata, cfg: Config, construct_edge_index=True, verbose=True):
    preprocessor = PreProcessor(cfg.n_pcs)
    preprocessor.fit_transform(adata)
    data = adata_to_pyg(adata, emb_key=cfg.emb_key)

    # for debugging preconstruct edge index
    if construct_edge_index:
        data.edge_index = construct_graph(
            data.pos,
            knn=cfg.knn,
            radius=cfg.radius,
            delaunay=cfg.delaunay,
            complete=cfg.complete,
            verbose=verbose,
        )
    return data


class PreProcessor:
    def __init__(self, n_pcs=50):
        self.pca = PCA(n_components=n_pcs)
        self.scaler = StandardScaler()

    def fit(self, adata):
        X_orig = adata.X.copy()
        adata.X = npc.utils.to_numpy(adata.X)

        # calc median
        self.median = np.median(adata.X.sum(axis=1), axis=0)
        sc.pp.normalize_total(adata, target_sum=self.median)

        # log1p
        adata.X = np.log1p(adata.X)

        # fit standard scaler
        adata.X = self.scaler.fit_transform(adata.X)

        # fit pca
        self.pca.fit(adata.X)

        # restore original data
        adata.X = X_orig

    def transform(self, adata):
        X_orig = adata.X.copy()
        adata.X = npc.utils.to_numpy(adata.X)

        # run transformations
        sc.pp.normalize_total(adata, target_sum=self.median)
        adata.X = np.log1p(adata.X)
        adata.X = self.scaler.transform(adata.X)
        adata.obsm["X_pca"] = self.pca.transform(adata.X)

        # restore original data
        adata.X = X_orig
        return adata

    def fit_transform(self, adata):
        self.fit(adata)
        return self.transform(adata)


def adata_to_pyg(adata, emb_key="X_pca", pos_key="spatial"):
    if emb_key is None:
        x = None
    elif emb_key == "X":
        x = npc.utils.to_torch(adata.X).float()
    else:
        x = npc.utils.to_torch(adata.obsm[emb_key]).float()

    pos = npc.utils.to_torch(adata.obsm[pos_key]).float()
    data = pyg.data.Data(x=x, pos=pos, pos_init=None)
    return data
