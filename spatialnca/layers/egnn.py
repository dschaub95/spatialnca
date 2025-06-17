from torch_geometric.nn import MessagePassing
import torch
import torch.nn as nn
from torch_scatter import scatter
from spatialnca.layers.mlp import SimpleMLP
from spatialnca.config import Config


class EGNNLayer(MessagePassing):
    def __init__(
        self,
        cfg: Config,
        **kwargs,
    ):
        super().__init__(aggr=cfg.aggr)
        msg_dim = cfg.msg_dim if cfg.msg_dim is not None else cfg.emb_dim
        self.normalize_diff = cfg.normalize_diff
        self.aggr_pos = cfg.aggr_pos
        self.max_scale = cfg.max_coord_upd_norm
        self.scale_by_dist = cfg.scale_by_dist

        self.mlp_msg = SimpleMLP(
            in_channels=cfg.emb_dim * 2 + 1,
            out_channels=msg_dim,
            hidden_channels=cfg.hidden_dim,
            n_layers=cfg.n_layers_msg,
            bias=cfg.bias,
            act=cfg.act,
            **kwargs,
        )
        self.mlp_upd = SimpleMLP(
            in_channels=cfg.emb_dim + msg_dim,
            out_channels=cfg.emb_dim,
            hidden_channels=cfg.hidden_dim,
            n_layers=cfg.n_layers_upd,
            bias=cfg.bias,
            act=cfg.act,
            **kwargs,
        )
        self.mlp_pos = nn.Sequential(
            SimpleMLP(
                in_channels=msg_dim,
                out_channels=1,
                hidden_channels=cfg.hidden_dim,
                n_layers=cfg.n_layers_pos,
                plain_last=True,  # allow for negative values
                bias=cfg.bias,
                act=cfg.act,
                **kwargs,
            ),
            nn.Tanh() if cfg.normalize_diff else nn.Identity(),
        )

        self.kernel = None
        if cfg.kernel_fn is not None:
            if cfg.kernel_fn == "gaussian":
                self.kernel = GaussianKernel(
                    sigma=cfg.kernel_kwargs.get("sigma", 0.0),
                    learnable=cfg.kernel_kwargs.get("learnable", True),
                )
            elif cfg.kernel_fn == "sigmoid":
                self.kernel = SigmoidKernel(
                    d1=cfg.kernel_kwargs.get("d1", 0.0),
                    d2=cfg.kernel_kwargs.get("d2", 1.0),
                    learnable=cfg.kernel_kwargs.get("learnable", True),
                )
            else:
                raise ValueError(f"Kernel function {cfg.kernel_fn} not supported")

    def forward(self, h, pos, edge_index, edge_attr=None):
        # compute initial feature transfo
        out = self.propagate(edge_index, h=h, x=pos, edge_attr=edge_attr)
        return out

    def message(self, h_i, h_j, x_i, x_j, edge_attr):
        diff = x_j - x_i
        # diff.shape = (num_edges, 2)

        # calculate standard euclidean distance
        dist = torch.norm(diff, p=2, dim=-1, keepdim=True)
        dist_kernel = self.kernel(dist) if self.kernel else dist

        msg = torch.cat([h_i, h_j, dist_kernel], dim=-1)

        # scale message by distance kernel if specified to effectively use a soft radius
        msg = (
            self.mlp_msg(msg) * dist_kernel if self.scale_by_dist else self.mlp_msg(msg)
        )

        # intuitively determines in which direction to move
        # might be improvable by using the message and the aggregated message
        # to get a full picture before determining what to focus on
        if self.max_scale is None:
            diff_scaler = self.mlp_pos(msg)
        else:
            c = (1 / torch.clamp(dist, min=1e-6)) * self.max_scale
            diff_scaler = smooth_saturating(self.mlp_pos(msg), c)
            # diff_scaler.shape = (num_edges, 1)

        # set norm to 1 --> network scaling will be off
        if self.normalize_diff:
            diff = diff / torch.clamp(dist.detach(), min=1e-6)

        diff_scaled = diff * diff_scaler

        # further scale by distance kernel if specified
        # to ensure far away nodes have limited impact
        diff_scaled = diff_scaled * dist_kernel if self.scale_by_dist else diff_scaled

        return msg, diff_scaled

    def aggregate(self, inputs, index):
        node_msg, pos_msg = inputs
        msg_aggr = scatter(node_msg, index, dim=self.node_dim, reduce=self.aggr)
        x_aggr = scatter(pos_msg, index, dim=self.node_dim, reduce=self.aggr_pos)
        return msg_aggr, x_aggr

    def update(self, aggr_out, h):
        msg_aggr, pos_upd = aggr_out
        h_upd = torch.cat([h, msg_aggr], dim=-1)
        h_upd = self.mlp_upd(h_upd)
        return h_upd, pos_upd


def smooth_saturating(x: torch.Tensor, c: float | torch.Tensor):
    # https://www.desmos.com/calculator/llsxsxolcl
    return c * torch.tanh(x / c)


class GaussianKernel(nn.Module):
    def __init__(self, sigma=0.0, learnable: bool = True):
        super().__init__()
        self.sigma = nn.Parameter(torch.ones(1) * sigma) if learnable else sigma

    def forward(self, x):
        # https://www.desmos.com/calculator/olo9fetyvt
        return torch.exp(-self.sigma * x**2)


class SigmoidKernel(nn.Module):
    def __init__(self, d1: float = 0.0, d2: float = 1.0, learnable: bool = True):
        super().__init__()
        self.d1 = nn.Parameter(torch.ones(1) * d1) if learnable else d1
        self.d2 = nn.Parameter(torch.ones(1) * d2) if learnable else d2

    def forward(self, dist):
        # https://www.desmos.com/calculator/jfbemb23hc
        exp = self.d2 * (dist - self.d1)
        return torch.sigmoid(-exp)
