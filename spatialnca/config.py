import torch
import numpy as np
from dataclasses import dataclass


class BaseConfig:
    def to_dict(self):
        return self.__dict__

    def __sub__(self, other: "BaseConfig") -> dict:
        """
        Returns a dictionary containing the entries that differ between two configs.
        Only compares entries that exist in both configs.

        Args:
            other: Config object to subtract from self

        Returns:
            Dictionary with entries that differ, showing value from self
        """
        dict1 = self.to_dict()
        dict2 = other.to_dict()

        # Find common keys
        common_keys = [k for k in dict1.keys() if k in dict2.keys()]

        # Compare values for common keys
        diffs = {}
        for key in common_keys:
            if dict1[key] != dict2[key]:
                diffs[key] = f"{dict2[key]}->{dict1[key]}"

        return diffs


@dataclass
class Config(BaseConfig):
    seed: int = 42

    # data
    path: str = "data/Zhuang-ABCA-3.003_slice.h5ad"
    n_pcs: int = 50
    emb_key: str | None = "X_pca"  # None to use learnable embedding per node

    # training
    n_epochs: int = 10000
    n_steps: int = 5
    lr: float = 1e-3
    warmup_iters: int = 0
    clip_value: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 1
    reinit_interval: int = np.inf
    pos_init_fn: str = "gaussian"
    pos_init_kwargs: dict | None = None
    weight_decay: float = 0.0
    dynamic_edges: bool = False
    n_static_warmup_steps: int = 0
    use_orig_graph: bool = True
    edge_update_steps: int = 1  # update the edge index every n_steps
    watch: bool = False
    plot_init_pos: bool = False

    # model
    emb_dim: int = 32
    hidden_dim: int = 128
    n_layers_msg: int = 2
    n_layers_upd: int = 2
    n_layers_pos: int = 2
    use_attn: bool = False
    aggr: str = "sum"
    aggr_pos: str = "mean"
    msg_dim: int | None = None
    norm: str | None = None
    bounds: tuple[float, float] | None = None
    skip_connections: bool = True
    add_init: bool = False
    knn: int = 10
    radius: float | None = None
    delaunay: bool = False
    complete: bool = False
    bias: bool = True
    act: str = "silu"
    gpt2_weight_init: bool = False
    intm_loss: bool = False  # calc loss after each step
    normalize_diff: bool = False
    lr_decay: bool = True
    max_coord_upd_norm: float | None = None
    kernel_fn: str | None = None
    kernel_kwargs: dict | None = None
    scale_by_dist: bool = False
    alpha_decay: bool = False
    node_norm_type: str | None = None
    node_norm_cap: float | None = None
