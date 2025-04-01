from torch_geometric.nn import MessagePassing
import torch
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

        self.aggr_pos = cfg.aggr_pos

        self.mlp_msg = SimpleMLP(
            in_channels=cfg.emb_dim * 2 + 1,
            out_channels=msg_dim,
            hidden_channels=cfg.hidden_dim,
            n_layers=cfg.n_layers_msg,
            bias=cfg.bias,
            **kwargs,
        )
        self.mlp_upd = SimpleMLP(
            in_channels=msg_dim * 2,
            out_channels=cfg.emb_dim,
            hidden_channels=cfg.hidden_dim,
            n_layers=cfg.n_layers_upd,
            bias=cfg.bias,
            **kwargs,
        )
        self.mlp_pos = SimpleMLP(
            in_channels=msg_dim,
            out_channels=1,
            hidden_channels=cfg.hidden_dim,
            n_layers=cfg.n_layers_pos,
            plain_last=True,  # allow for negative values
            bias=cfg.bias,
            **kwargs,
        )

    def forward(self, h, pos, edge_index, edge_attr=None):
        # compute initial feature transfo
        out = self.propagate(edge_index, h=h, x=pos, edge_attr=edge_attr)
        return out

    def message(self, h_i, h_j, x_i, x_j, edge_attr):
        diff = x_j - x_i
        dist = torch.norm(diff, p=2, dim=-1, keepdim=True)

        msg = torch.cat([h_i, h_j, dist], dim=-1)
        msg = self.mlp_msg(msg)

        # intuitively determines in which direction to move
        # might be imporvable by using the message and the aggregated message
        # to get a full picture before determinign waht do focus on
        diff_scaled = diff * self.mlp_pos(msg)

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
