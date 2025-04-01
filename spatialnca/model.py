import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MLP, MessagePassing
import torch_geometric as pyg
from torch_scatter import scatter

from spatialnca.layers.mlp import SimpleMLP
from spatialnca.layers.egnn import EGNNLayer


class SpatialNCA(nn.Module):
    def __init__(
        self,
        input_dim,
        emb_dim,
        knn=10,
        radius=None,
        reinit=False,
        skip_connections=True,
        bounds=None,
        use_fixed_emb=False,  # for debugging
        fixed_edge_index=False,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.knn = knn
        self.radius = radius
        self.reinit = reinit
        self.skip_connections = skip_connections
        self.bounds = bounds
        self.mpnn = EGNNLayer(emb_dim, **kwargs)
        self.mlp_emb = SimpleMLP(
            in_channels=input_dim,
            out_channels=emb_dim,
            n_layers=2,
            plain_last=True,
            **kwargs,
        )
        self.use_fixed_emb = use_fixed_emb
        self.fixed_emb = None
        self.fixed_edge_index = fixed_edge_index

    def init_fixed_emb(self, n_cells):
        if self.fixed_emb is None:
            self.fixed_emb = nn.Parameter(
                torch.randn(n_cells, self.emb_dim, device=self.device)
            )

    def forward(self, h, pos, edge_index, h_init, edge_attr=None):
        # add initial node features
        if self.reinit:
            h = h + h_init

        # add residual connection
        h_update, pos_update = self.mpnn(h, pos, edge_index)
        if self.skip_connections:
            h = h + h_update
        else:
            h = h_update
        pos = pos + pos_update
        return h, pos

    def rollout(self, x, pos, n_steps, h=None, edge_index=None, loss_fn=None):
        tot_loss = 0
        n_steps = max(1, n_steps)

        edge_index = self.init_edge_index(pos) if edge_index is None else edge_index

        # embed the input features and store them
        if self.use_fixed_emb:
            self.init_fixed_emb(x.shape[0])
            self.h_init = self.fixed_emb
        else:
            self.h_init = self.mlp_emb(x)

        # start from the initial embedding if no (updated) h is provided
        h = self.h_init if h is None else h

        # run for multiple steps
        for _ in range(n_steps):
            h, pos, edge_index = self.step(h, pos, edge_index)

            # compute intermediate loss (optional)
            if loss_fn is not None:
                loss = loss_fn(pos)
                tot_loss += loss

        if loss_fn is not None:
            return h, pos, edge_index, tot_loss
        else:
            return h, pos, edge_index

    def step(self, h, pos, edge_index):
        h, pos_new = self.forward(h, pos, edge_index, h_init=self.h_init)

        # TODO track the change in pos per node (max, mean, median might be interesting)
        # diff = torch.norm(pos_new - pos, p=2, dim=-1)
        # mean_diff = diff.mean().item()
        # max_diff = diff.max().item()
        # min_diff = diff.min().item()
        # median_diff = torch.median(diff).item()
        
        pos = pos_new

        # confine pos to bound range
        if self.bounds is not None:
            pos = torch.clamp(pos, min=self.bounds[0], max=self.bounds[1])

        # update the edge index based on the new positions
        if not self.fixed_edge_index:
            edge_index = self.update_edge_index(pos, edge_index)

        return h, pos, edge_index

    def update_edge_index(self, pos, edge_index=None, batch=None):
        # Ensure at least one of radius or k-NN is specified
        assert self.radius or self.knn, "Either radius or knn must be specified"

        combined_edge_index = []

        # Compute radius graph if radius is specified
        if self.radius:
            radius_edge_index = pyg.nn.radius_graph(
                pos,
                r=self.radius,
                loop=False,
                batch=batch,
                flow="source_to_target",
                max_num_neighbors=64,
            )
            combined_edge_index.append(radius_edge_index)

        # Compute k-NN graph if k is specified
        if self.knn:
            knn_edge_index = pyg.nn.knn_graph(
                pos, k=self.knn, loop=False, flow="source_to_target", batch=batch
            )
            combined_edge_index.append(knn_edge_index)

        if len(combined_edge_index) > 1:
            combined_edge_index = torch.cat(combined_edge_index, dim=1)
            # Remove duplicate edges
            combined_edge_index = torch.unique(combined_edge_index, dim=1)
        else:
            # Use the single available edge index
            combined_edge_index = combined_edge_index[0]
        return combined_edge_index

    def init_edge_index(self, pos, **kwargs):
        return self.update_edge_index(pos, **kwargs)

    @property
    def device(self):
        return next(self.parameters()).device


def construct_graph(pos, radius=None, knn=None, batch=None):
    if radius is not None:
        edge_index = pyg.nn.radius_graph(
            pos, r=radius, loop=True, flow="source_to_target", batch=batch
        )
    elif knn is not None:
        edge_index = pyg.nn.knn_graph(
            pos, k=knn, loop=True, flow="source_to_target", batch=batch
        )
    else:
        raise ValueError("Either radius or knn must be specified")
    return edge_index
