import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import math
import wandb
from pprint import pprint
import matplotlib.pyplot as plt
import scanpy as sc

from spatialnca.model import SpatialNCA
from spatialnca.spatial import uniform_point_cloud, sunflower_points, random_walk
from spatialnca.config import Config
from spatialnca.utils import construct_graph, seed_everything
from spatialnca.data import prepare_data


def train(cfg: Config, print_cfg: bool = False):
    with wandb.init(project="spatialnca", config=cfg.to_dict()) as run:
        cfg = Config(**run.config)
        if print_cfg:
            pprint(cfg.to_dict())

        adata = sc.read_h5ad(cfg.path)

        # seed everything
        seed_everything(cfg.seed)

        data = prepare_data(adata, cfg)

        spnca = SpatialNCA(
            data.num_features,
            cfg,
        )

        if cfg.watch:
            wandb.watch(spnca, log="all")

        trainer = Trainer(model=spnca, cfg=cfg)

        trainer.train(data)

    return trainer


class Trainer:
    def __init__(
        self,
        model: SpatialNCA,
        cfg: Config,
    ):
        self.cfg = cfg
        self.n_epochs = cfg.n_epochs
        self.batch_size = cfg.batch_size
        self.lr = cfg.lr
        self.clip_value = cfg.clip_value
        self.reintv = cfg.reinit_interval
        self.device = cfg.device
        self.pos_init_fn = cfg.pos_init_fn
        self.pos_init_kwargs = cfg.pos_init_kwargs or {}
        self.intm_loss = cfg.intm_loss
        self.weight_decay = cfg.weight_decay
        self.dynamic_edges = cfg.dynamic_edges
        self.n_static_warmup_steps = cfg.n_static_warmup_steps

        self.model = model
        self.step_sampler = StepSampler(cfg.n_steps)
        self.lr_scheduler = (
            LearningRateScheduler(
                lr=cfg.lr,
                warmup_iters=cfg.warmup_iters,
                lr_decay_iters=cfg.n_epochs,
                min_lr=cfg.lr / 10,
            )
            if cfg.lr_decay
            else None
        )
        self.logger = WandbLogger()

    def setup_training(self, data):
        # TODO make this batch aware for later

        # transfer to device
        data = data.to(self.device)
        self.data = data
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

    def setup_optimizer(self):
        # setup optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def train(self, data):
        self.setup_training(data)

        prog_bar = tqdm(range(self.n_epochs), desc="Training")

        for epoch in prog_bar:
            # reinit node positions regularly
            if epoch % self.reintv == 0:
                self.init_node_positions(data)

            if self.lr_scheduler is not None:
                lr = self.lr_scheduler.get_lr(epoch)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr

            # train one epoch
            loss = self.train_one_epoch(
                data,
                dynamic_edges=self.dynamic_edges
                if epoch >= self.n_static_warmup_steps
                else False,
            )

            # log metrics
            self.logger.log({"epoch": epoch, "train_loss": loss.item()})

            # dynamically update progress bar
            prog_bar.set_postfix(
                {k: v for k, v in self.logger.history[-1].items() if k != "epoch"}
            )

    def train_one_epoch(self, data, dynamic_edges=False):
        # sample number of steps
        n_steps = self.step_sampler.sample()

        # rollout
        h, pos, edge_index, loss = self.model.rollout(
            x=data.x,
            pos=data.pos_init,
            n_steps=n_steps,
            edge_index=data.edge_index
            if self.cfg.use_orig_graph
            else data.edge_index_init,
            loss_fn=self.compute_loss if self.intm_loss else None,
            dynamic_edges=dynamic_edges,
        )

        # compute loss if not provided
        loss = self.compute_loss(pos) if loss is None else loss

        # backprop
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value)
        self.optimizer.step()

        # check for NaN loss
        assert not pd.isna(loss.item()), (
            f"Loss is NaN:\n"
            f"loss: {loss}\n"
            f"pos: {pos}\n"
            f"edge_index: {edge_index}\n"
            f"data.pos_init: {data.pos_init}\n"
            f"data.pos: {data.pos}"
        )
        return loss

    def compute_loss(self, pos: torch.Tensor):
        dists_pred = torch.norm(
            pos[self.edge_index_full[0]] - pos[self.edge_index_full[1]], p=2, dim=-1
        )
        loss = F.mse_loss(dists_pred, self.dists_true)
        return loss

    def plot_history(self, log_scale=True, title=None, save_path=None):
        df = pd.DataFrame(self.logger.history)
        plt.plot(df["train_loss"])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        if log_scale:
            plt.yscale("log")
        if title is not None:
            plt.title(title)
        if save_path is not None:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.show()

    def init_node_positions(self, data):
        max_dist = self.dists_true.max().cpu().item()
        if self.pos_init_fn == "gaussian":
            data.pos_init = self.pos_init_kwargs.get("scale", 1) * torch.randn(
                data.pos.shape[0], 2, device=self.device
            )
        elif self.pos_init_fn == "uniform":
            # calculate based on number of nodes
            data.pos_init = torch.tensor(
                uniform_point_cloud(data.num_nodes, max_dist / 4),
                device=self.device,
            )
        elif self.pos_init_fn == "sunflower":
            data.pos_init = torch.tensor(
                sunflower_points(data.num_nodes, median_dist=0.005),
                device=self.device,
                dtype=torch.float32,
            )
        elif self.pos_init_fn == "random_walk":
            # calc median dist
            if "max_displacement" in self.pos_init_kwargs:
                max_dist = self.pos_init_kwargs["max_displacement"]
            else:
                edge_index = construct_graph(data.pos, delaunay=True)
                dists = torch.norm(
                    data.pos[edge_index[0]] - data.pos[edge_index[1]], p=2, dim=-1
                )
                max_dist = dists.median().detach().cpu().item()
            data.pos_init = random_walk(
                data.pos,
                num_steps=self.pos_init_kwargs.get("num_steps", 1),
                max_displacement=max_dist,
                progress_bar=False,
            )
        else:
            raise ValueError(f"Unknown pos_init_fn: {self.pos_init_fn}")
        
        if self.cfg.plot_init_pos:
            x, y = data.pos_init.detach().cpu().numpy().T
            fig = plt.figure(figsize=(5, 5))
            plt.scatter(x, y, s=5)
            plt.gca().set_aspect("equal")
            plt.title("Initial positions")
            self.logger.log_plot(fig, "init_pos")

        # directly construct the initial edge index to avoid recomputing it in the rollout
        if self.cfg.complete:
            # prevent recomputing the edge index to save memory
            data.edge_index_init = data.edge_index
        else:
            data.edge_index_init = construct_graph(
                data.pos_init,
                radius=self.cfg.radius,
                knn=self.cfg.knn,
                delaunay=self.cfg.delaunay,
                batch=data.batch,
            )


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


class LearningRateScheduler:
    def __init__(self, lr, warmup_iters, lr_decay_iters, min_lr):
        self.lr = lr
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters
        self.min_lr = min_lr

    def get_lr(self, it):
        # 1) linear warmup for warmup_iters steps
        if it < self.warmup_iters:
            return self.lr * (it + 1) / (self.warmup_iters + 1)
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.lr_decay_iters:
            return self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.warmup_iters) / (
            self.lr_decay_iters - self.warmup_iters
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return self.min_lr + coeff * (self.lr - self.min_lr)


class WandbLogger:
    def __init__(self):
        self.history = []

    def log(self, metrics: dict, **kwargs):
        self.history.append(metrics)
        if wandb.run is not None:
            wandb.run.log(metrics, **kwargs)
        
    def log_plot(self, plot, name):
        if wandb.run is not None:
            wandb.run.log({name: plot})
