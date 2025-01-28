import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt

from spatialnca.model import SpatialNCA


class Trainer:
    def __init__(
        self,
        model: SpatialNCA,
        n_epochs=1000,
        n_steps=10,
        lr=1e-3,
        clip_value=1.0,
        reinit_interval=np.inf,
        device="cuda",
    ):
        self.n_epochs = n_epochs
        self.lr = lr
        self.clip_value = clip_value
        self.reintv = reinit_interval
        self.device = device

        self.model = model
        self.step_sampler = StepSampler(n_steps)

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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # setup history
        self.history = {"train_loss": []}

    def train(self, data):
        self.setup_training(data)

        for epoch in tqdm(range(self.n_epochs)):
            # reinit node positions regularly
            if epoch % self.reintv == 0:
                data.pos_init = torch.randn(data.pos.shape[0], 2, device=self.device)
            self.train_one_epoch(data)

    def train_one_epoch(self, data):
        n_steps = self.step_sampler.sample()
        h, pos, edge_index = self.model.rollout(
            x=data.x, pos=data.pos_init, n_steps=n_steps, edge_index=data.edge_index
        )
        dists_pred = torch.norm(
            pos[self.edge_index_full[0]] - pos[self.edge_index_full[1]], p=2, dim=-1
        )
        loss = F.mse_loss(dists_pred, self.dists_true)
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_value)
        self.optimizer.step()

        self.history["train_loss"].append(loss.item())

    def plot_history(self):
        plt.plot(self.history["train_loss"])
        plt.show()


class Tester:
    def __init__(self):
        pass

    def test(self):
        pass


class StepSampler:
    def __init__(self, n_steps=None, min_steps=5, max_steps=10):
        self.n_steps = n_steps

    def sample(self):
        return self.n_steps
