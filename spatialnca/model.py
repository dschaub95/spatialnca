import torch
import torch.nn as nn
from tqdm.auto import tqdm

from spatialnca.layers.mlp import SimpleMLP
from spatialnca.layers.egnn import EGNNLayer
from spatialnca.config import Config
from spatialnca.utils import construct_graph, isna


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
        self.delaunay = cfg.delaunay
        self.add_init = cfg.add_init
        self.skip_connections = cfg.skip_connections
        self.bounds = cfg.bounds
        self.edge_update_steps = cfg.edge_update_steps

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

    def rollout(
        self,
        x,
        pos,
        n_steps,
        h=None,
        edge_index=None,
        loss_fn=None,
        return_evolution=False,
        prog_bar=False,
        dynamic_edges=False,
    ):
        assert n_steps > 0, "n_steps must be greater than 0"

        # create edge index if not provided
        if edge_index is None:
            edge_index = self.init_edge_index(pos)

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
        loss = 0 if loss_fn is not None else None
        # track results per step if return_evolution is True
        states = [save_state(h, pos, edge_index, loss, 0)]
        for i in tqdm(
            range(n_steps), total=n_steps, desc="Rollout", disable=not prog_bar
        ):
            h, pos, edge_index = self.step(h, pos, edge_index)

            # update the edge index based on the new positions
            if dynamic_edges and (i + 1) % self.edge_update_steps == 0:
                edge_index = self.update_edge_index(pos, edge_index=edge_index)

            if isna(h):
                raise ValueError(f"h contains NaNs in step {i}")

            if isna(pos):
                raise ValueError(f"pos contains NaNs in step {i}")

            # compute mean intermediate loss (optional)
            if loss_fn is not None:
                loss += loss_fn(pos) * (i + 1) / n_steps

            if return_evolution:
                states.append(save_state(h, pos, edge_index, loss, i + 1))

        if return_evolution:
            return states
        else:
            return h, pos, edge_index, loss

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

        return h, pos, edge_index

    def update_edge_index(self, pos, edge_index=None, batch=None):
        edge_index = construct_graph(
            pos,
            radius=self.radius,
            knn=self.knn,
            delaunay=self.delaunay,
            batch=batch,
        )
        return edge_index

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


def save_state(h, pos, edge_index, loss, step):
    return {
        "h": h.detach().cpu(),
        "pos": pos.detach().cpu(),
        "edge_index": edge_index.detach().cpu(),
        "loss": loss.detach().cpu() if loss is not None else None,
        "step": step,
    }
