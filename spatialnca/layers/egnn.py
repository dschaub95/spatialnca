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
            in_channels=msg_dim * 2,
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

        self.kernel = (
            None
            if cfg.sigma_init is None
            else GaussianKernel(sigma_init=cfg.sigma_init)
        )

    def forward(self, h, pos, edge_index, edge_attr=None):
        # compute initial feature transfo
        out = self.propagate(edge_index, h=h, x=pos, edge_attr=edge_attr)
        return out

    def message(self, h_i, h_j, x_i, x_j, edge_attr):
        diff = x_j - x_i
        dist = torch.norm(diff, p=2, dim=-1, keepdim=True)
        dist_kernel = self.kernel(dist) if self.kernel else dist

        # set norm to 1 --> network scaling will be off
        if self.normalize_diff:
            diff = diff / torch.clamp(dist.detach(), min=1e-6)

        msg = torch.cat([h_i, h_j, dist_kernel], dim=-1)
        # scale message by distance kernel if specified to effectively use a soft radius
        msg = self.mlp_msg(msg) * dist_kernel if self.kernel else self.mlp_msg(msg)

        # intuitively determines in which direction to move
        # might be imporvable by using the message and the aggregated message
        # to get a full picture before determinign waht do focus on
        if self.max_scale is None:
            diff_scaler = self.mlp_pos(msg)
        else:
            c = (1 / torch.clamp(dist, min=1e-6)) * self.max_scale
            diff_scaler = smooth_saturating(self.mlp_pos(msg), c)
        diff_scaled = diff * diff_scaler

        # further scale by distance kernel if specified
        # to ensure far away nodes have limited impact
        diff_scaled = diff_scaled * dist_kernel if self.kernel else diff_scaled

        return msg, diff_scaled

    def aggregate(self, inputs, index):
        node_msg, pos_msg = inputs
        h_aggr = scatter(node_msg, index, dim=self.node_dim, reduce=self.aggr)
        x_aggr = scatter(pos_msg, index, dim=self.node_dim, reduce=self.aggr_pos)
        return h_aggr, x_aggr

    def update(self, aggr_out, h):
        h_aggr, pos_upd = aggr_out
        h_upd = torch.cat([h, h_aggr], dim=-1)
        h_upd = self.mlp_upd(h_upd)
        return h_upd, pos_upd


def smooth_saturating(x: torch.Tensor, c: float | torch.Tensor):
    return c * torch.tanh(x / c)


class GaussianKernel(nn.Module):
    def __init__(self, sigma_init=0.0):
        super().__init__()
        self.sigma = torch.nn.Parameter(torch.tensor(sigma_init))

    def forward(self, x):
        return torch.exp(-self.sigma * x**2)
