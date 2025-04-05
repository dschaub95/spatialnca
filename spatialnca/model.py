import torch
import torch.nn as nn
import torch_geometric as pyg

from spatialnca.layers.mlp import SimpleMLP
from spatialnca.layers.egnn import EGNNLayer
from spatialnca.config import Config


class SpatialNCA(nn.Module):
    def __init__(
        self,
        input_dim,
        cfg: Config,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = cfg.emb_dim
        self.knn = cfg.knn
        self.radius = cfg.radius
        self.add_init = cfg.add_init
        self.skip_connections = cfg.skip_connections
        self.bounds = cfg.bounds

        self.mpnn = EGNNLayer(cfg)
        self.mlp_emb = SimpleMLP(
            in_channels=input_dim,
            out_channels=cfg.emb_dim,
            n_layers=2,
            plain_last=True,
            bias=cfg.bias,
            act=cfg.act,
            **kwargs,
        )
        self.fixed_emb = None
        self.fixed_edge_index = True

        # Initialize weights
        if cfg.gpt2_weight_init:
            self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def init_fixed_emb(self, n_cells):
        if self.fixed_emb is None:
            self.fixed_emb = nn.Parameter(
                torch.randn(n_cells, self.emb_dim, device=self.device)
            )

    def forward(self, h, pos, edge_index, h_init, edge_attr=None):
        # add initial node features
        h = h + h_init if self.add_init else h

        # add residual connection
        h_update, pos_update = self.mpnn(h, pos, edge_index)
        h = h + h_update if self.skip_connections else h_update
        pos = pos + pos_update
        return h, pos

    def rollout(self, x, pos, n_steps, h=None, edge_index=None, loss_fn=None):
        tot_loss = 0
        n_steps = max(1, n_steps)

        # create edge index if not provided, otherwise keep it fixed
        if edge_index is None:
            edge_index = self.init_edge_index(pos)
            self.fixed_edge_index = False
        else:
            self.fixed_edge_index = True

        # embed the input features and store them
        if x is None:
            self.init_fixed_emb(pos.shape[0])
            self.h_init = self.fixed_emb
        else:
            assert isinstance(x, torch.Tensor)
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
                loop=True,
                batch=batch,
                flow="source_to_target",
                max_num_neighbors=64,
            )
            combined_edge_index.append(radius_edge_index)

        # Compute k-NN graph if k is specified
        if self.knn:
            knn_edge_index = pyg.nn.knn_graph(
                pos, k=self.knn, loop=True, flow="source_to_target", batch=batch
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

    def get_num_params(self):
        """
        Return the number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
