import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MLP, MessagePassing
import torch_geometric as pyg
from torch_scatter import scatter


class SpatialNCA(nn.Module):
    def __init__(self, input_dim, emb_dim, knn=10, reinit=False, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.knn = knn
        self.reinit = reinit

        self.mpnn = MPNNLayer(emb_dim, **kwargs)
        self.mlp_emb = SimpleMLP(
            in_channels=input_dim,
            out_channels=emb_dim,
            hidden_channels=emb_dim,
            n_layers=1,
            plain_last=True,
        )

    def forward(self, h, pos, edge_index, h_init, edge_attr=None):
        # add initial node features
        if self.reinit:
            h = h + h_init

        # add residual connection
        h_update, pos_update = self.mpnn(h, pos, edge_index)
        h = h + h_update
        pos = pos + pos_update
        return h, pos

    def rollout(self, x, pos, n_steps, h=None, edge_index=None, loss_fn=None):
        tot_loss = 0
        n_steps = max(1, n_steps)

        edge_index = self.init_edge_index(pos) if edge_index is None else edge_index

        # embed the input features and store them
        self.h_init = self.mlp_emb(x)

        if h is None:
            h = self.h_init

        # run for multiple steps
        for _ in range(n_steps):
            h, pos = self.forward(h, pos, edge_index, h_init=self.h_init)

            # confine pos to [0, 1] range
            pos = torch.clamp(pos, min=0, max=1)

            # update the edge index based on the new positions
            edge_index = self.update_edge_index(pos, edge_index)

            # compute intermediate loss (optional)
            if loss_fn is not None:
                loss = loss_fn(pos)
                tot_loss += loss

        if loss_fn is not None:
            return h, pos, edge_index, tot_loss
        else:
            return h, pos, edge_index

    def update_edge_index(self, pos, edge_index=None, batch=None):
        # this graph is directed, exclude self loops for now, automatically handles device
        edge_index = pyg.nn.knn_graph(
            pos, k=self.knn, loop=False, flow="source_to_target", batch=batch
        )
        return edge_index

    def init_edge_index(self, pos, **kwargs):
        return self.update_edge_index(pos, **kwargs)

    @property
    def device(self):
        return next(self.parameters()).device


class MPNNLayer(MessagePassing):
    def __init__(
        self,
        emb_dim,
        hidden_dim=128,
        n_layers_msg=2,
        n_layers_upd=1,
        n_layers_pos=2,
        aggr="sum",
        aggr_pos="mean",
        **kwargs,
    ):
        super().__init__(aggr=aggr)
        self.aggr_pos = aggr_pos

        self.mlp_msg = SimpleMLP(
            in_channels=emb_dim * 2 + 1,
            out_channels=emb_dim,
            hidden_channels=hidden_dim,
            n_layers=n_layers_msg,
            **kwargs,
        )
        self.mlp_upd = SimpleMLP(
            in_channels=emb_dim * 2,
            out_channels=emb_dim,
            hidden_channels=hidden_dim,
            n_layers=n_layers_upd,
            **kwargs,
        )
        self.mlp_pos = SimpleMLP(
            in_channels=emb_dim,
            out_channels=1,
            hidden_channels=hidden_dim,
            n_layers=n_layers_pos,
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


class SimpleMLP(MLP):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels=None,
        n_layers=1,
        act="relu",
        dropout=0.0,
        norm="batch_norm",
        plain_last=False,
        **kwargs,
    ):
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        super().__init__(
            channel_list=None,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_layers=n_layers,
            act=act,
            dropout=dropout,
            norm=norm,
            plain_last=plain_last,
            **kwargs,
        )
