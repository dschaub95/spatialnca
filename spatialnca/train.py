import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from spatialnca.model import SpatialNCA
from spatialnca.spatial import uniform_point_cloud
from spatialnca.config import Config


class Trainer:
    def __init__(
        self,
        model: SpatialNCA,
        cfg: Config,
    ):
        self.n_epochs = cfg.n_epochs
        self.batch_size = cfg.batch_size
        self.lr = cfg.lr
        self.clip_value = cfg.clip_value
        self.reintv = cfg.reinit_interval
        self.device = cfg.device

        self.model = model
        self.step_sampler = StepSampler(cfg.n_steps)

    def setup_training(self, data):
        # TODO make this batch aware for later

        # transfer to device
        data = data.to(self.device)
        self.model = self.model.to(self.device)

        # create full edge index for training loss calculation
        num_nodes = data.num_nodes
        edge_index_full = (
            torch.ones(num_nodes, num_nodes).tril(-1).nonzero().T.to(self.device)
        )
        dists_true = torch.norm(
            data.pos[edge_index_full[0]] - data.pos[edge_index_full[1]], p=2, dim=-1
        )

        self.edge_index_full = edge_index_full
        self.dists_true = dists_true

        # setup optimizer
        self.setup_optimizer()

        # setup history
        self.history = []

    def setup_optimizer(self):
        # setup optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self, data):
        self.setup_training(data)

        prog_bar = tqdm(range(self.n_epochs), desc="Training")

        for epoch in prog_bar:
            # reinit node positions regularly
            if epoch % self.reintv == 0:
                self.init_node_positions(data)

            # train one epoch
            loss = self.train_one_epoch(data)

            # store history
            self.history.append({"epoch": epoch, "train_loss": loss.item()})

            # dynamically update progress bar
            prog_bar.set_postfix(
                {k: v for k, v in self.history[-1].items() if k != "epoch"}
            )

    def train_one_epoch(self, data):
        # sample number of steps
        n_steps = self.step_sampler.sample()

        # rollout
        h, pos, edge_index = self.model.rollout(
            x=data.x, pos=data.pos_init, n_steps=n_steps, edge_index=data.edge_index
        )

        # compute loss
        dists_pred = torch.norm(
            pos[self.edge_index_full[0]] - pos[self.edge_index_full[1]], p=2, dim=-1
        )
        loss = F.mse_loss(dists_pred, self.dists_true)

        # backprop
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_value)
        self.optimizer.step()

        return loss

    def plot_history(self, log_scale=True, title=None, save_path=None):
        df = pd.DataFrame(self.history)
        plt.plot(df["train_loss"])
        if log_scale:
            plt.yscale("log")
        if title is not None:
            plt.title(title)
        if save_path is not None:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.show()

    def init_node_positions(self, data):
        data.pos_init = 1 * torch.randn(data.pos.shape[0], 2, device=self.device)
        # data.pos_init = torch.zeros(data.pos.shape[0], 2, device=self.device)

        # data.pos_init = torch.tensor(
        #     uniform_point_cloud(data.num_nodes, 0.15),
        #     device=self.device,
        # )

        # median_dist = self.dists_true.median()
        # data.pos_init = torch.tensor(
        #     uniform_point_cloud(data.num_nodes, median_dist / 2),
        #     device=self.device,
        # )
        # adapt the model radius to the scale of the data
        # self.model.radius = 2 * median_dist


class Tester:
    def __init__(self):
        pass

    def test(self):
        pass


class StepSampler:
    def __init__(self, n_steps=None, min_steps=None, max_steps=None, seed=42):
        self.n_steps = n_steps
        self.min_steps = min_steps
        self.max_steps = max_steps

        self.rng = np.random.default_rng(seed)

        assert n_steps is not None or (
            min_steps is not None and max_steps is not None
        ), "Either n_steps or min_steps and max_steps must be provided"

    def sample(self):
        if self.n_steps is not None:
            return self.n_steps
        else:
            return self.rng.integers(self.min_steps, self.max_steps + 1)
