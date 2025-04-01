import torch
import numpy as np
from dataclasses import dataclass


class BaseConfig:
    def to_dict(self):
        return self.__dict__


@dataclass
class Config(BaseConfig):
    seed: int = 42

    # data
    n_pcs: int = 50

    # training
    n_epochs: int = 10000
    n_steps: int = 5
    lr: float = 1e-3
    clip_value: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 1
    reinit_interval: int = np.inf

    # model
    emb_dim: int = 32
    hidden_dim: int = 128
    n_layers_msg: int = 2
    n_layers_upd: int = 2
    n_layers_pos: int = 2
    aggr: str = "sum"
    aggr_pos: str = "mean"
    msg_dim: int | None = None
    norm: str | None = None
    bounds: tuple[float, float] | None = None
    skip_connections: bool = True
    add_init: bool = False
    knn: int = 10
    radius: float | None = None
    bias: bool = True


# class ModelConfig(BaseConfig):
#     emb_dim: int = 32
#     hidden_dim: int = 1 * emb_dim
#     knn: int = 10
#     msg_dim: int | None = None
#     norm: str | None = None
#     use_fixed_emb: bool = True
#     use_fixed_edge_index: bool = True
#     bounds: tuple[float, float] | None = None


# class DataConfig(BaseConfig):
#     n_pcs: int = 50


# class TrainConfig(BaseConfig):
#     n_epochs: int = 5000
#     n_steps: int = 5
#     lr: float = 1e-3
#     clip_value: float = 1.0
#     device: str = "cuda" if torch.cuda.is_available() else "cpu"
