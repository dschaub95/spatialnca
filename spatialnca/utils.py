import numpy as np
import scipy
import networkx as nx
import torch_geometric as pyg
import torch
import matplotlib.pyplot as plt
import random
import seaborn as sns
import scanpy as sc
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
from matplotlib.animation import FFMpegWriter

from torch_geometric.transforms.delaunay import Delaunay
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


def grid2d(shape=(10, 10)):
    x = np.linspace(0, 1, shape[0])
    y = np.linspace(0, 1, shape[1])
    coords = np.stack(np.meshgrid(x, y), -1).reshape(-1, 2)
    return coords


def grid2d_graph(shape=(10, 10), radius=0.15, self_loops=False):
    coords = grid2d(shape).astype(np.float32)
    kdtree = scipy.spatial.KDTree(coords)
    dist_mat = kdtree.sparse_distance_matrix(kdtree, radius, p=2)

    edge_index, edge_weight = pyg.utils.from_scipy_sparse_matrix(dist_mat)

    if not self_loops:
        edge_index, edge_weight = pyg.utils.remove_self_loops(edge_index, edge_weight)

    edge_attr = edge_weight.unsqueeze(-1).to(torch.float32)
    data = pyg.data.Data(
        edge_index=edge_index, edge_attr=edge_attr, pos=torch.tensor(coords)
    )

    return data


def generate_grid_adata(shape=(10, 10)):
    coords = grid2d(shape)
    adata = sc.AnnData(
        X=np.random.randint(0, 1000, (coords.shape[0], 100)), obsm={"spatial": coords}
    )
    return adata


def to_networkx(edge_index):
    # transform the graph inside the adata object to a networkx graph
    edge_index = edge_index.detach().cpu().numpy()
    g = nx.Graph()
    g.add_edges_from(edge_index.T)
    return g


def plot_pyg(edge_index, pos=None, show_selfloops=False):
    if pos is not None:
        pos = pos.detach().cpu().numpy()

    g = to_networkx(edge_index)

    if not show_selfloops:
        g.remove_edges_from(nx.selfloop_edges(g))

    fig, ax = plt.subplots(figsize=(5, 5))
    nx.draw(g, pos=pos, ax=ax, node_size=10)
    plt.show()


def random_k_regular_graph(num_nodes, k, seed=42, device="cpu"):
    G = nx.random_regular_graph(d=k, n=num_nodes, seed=seed)
    edge_index = pyg.utils.from_networkx(G).edge_index
    edge_index = edge_index.to(device)
    return edge_index


def construct_graph(
    pos: torch.Tensor,
    radius=None,
    knn=None,
    complete=False,
    delaunay=False,
    batch=None,
    verbose=False,
):
    if complete:
        edge_index = complete_graph(pos.shape[0], batch=batch, device=pos.device)
    else:
        assert any([radius, knn, delaunay]), (
            "At least one of radius, knn, or delaunay must be provided"
        )
        edge_indices = []
        if radius is not None:
            edge_index = pyg.nn.radius_graph(
                pos, r=radius, loop=True, flow="source_to_target", batch=batch
            )
            edge_indices.append(edge_index)
        if knn is not None:
            edge_index = pyg.nn.knn_graph(
                pos, k=knn, loop=True, flow="source_to_target", batch=batch
            )
            edge_indices.append(edge_index)
        if delaunay:
            face = Delaunay()(Data(pos=pos.detach())).face
            row = torch.cat([face[0], face[1], face[2]])
            col = torch.cat([face[1], face[2], face[0]])
            edge_index = torch.stack([row, col], dim=0)
            edge_index = to_undirected(edge_index)
            edge_indices.append(edge_index)

        # remove duplicate edges
        if len(edge_indices) > 1:
            edge_index = torch.cat(edge_indices, dim=1)
            edge_index = torch.unique(edge_index, dim=1)
        else:
            edge_index = edge_indices[0]

    if verbose:
        print(
            f"Constructed graph with {pos.shape[0]} nodes and {edge_index.shape[1]} edges"
        )
    return edge_index


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)


def spatial_scatter(adata, color=None, pos_key="spatial", title=None, cmap=None):
    sns.set_theme(style="ticks", context="paper")
    x, y = adata.obsm[pos_key].T

    if color is not None:
        obs_data = adata.obs[color]
        unique_cats = obs_data.unique()
        n_cats = len(unique_cats)

        palette = sns.color_palette(
            "tab20" if n_cats > 10 else "tab10", len(unique_cats)
        )
        color_mapping = {category: palette[i] for i, category in enumerate(unique_cats)}

        colors = [color_mapping[e] for e in obs_data.values]
    else:
        colors = "blue"

    # Plot with subplots and clear axes
    # fig, ax = plt.subplots(figsize=(6, 6))
    fig, ax = plt.subplots()

    scatter = ax.scatter(x, y, c=colors, label=color)
    if color is not None:
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color=palette[i],
                linestyle="None",
                markersize=8,
                label=category,
            )
            for i, category in enumerate(unique_cats)
        ]
        ax.legend(
            handles=handles, title=color, loc="center left", bbox_to_anchor=(1, 0.5)
        )
    # Set equal scaling for the x and y axes
    # # ax.set_aspect("equal", adjustable="datalim")
    # # plt.tight_layout()
    # # Add axis labels and title
    # fig.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.1)  # Fine-tune padding
    ax.set_xlabel("X-coordinate")
    ax.set_ylabel("Y-coordinate")

    if title is not None:
        ax.set_title(title)
    plt.show()


def complete_graph(num_nodes, batch=None, device=None):
    if batch is None:
        edge_index = torch.cartesian_prod(
            torch.arange(num_nodes), torch.arange(num_nodes)
        ).T
    else:
        # TODO implement batch
        raise NotImplementedError("Batch not implemented yet")
    if device is not None:
        edge_index = edge_index.to(device)
    return edge_index


def set_color(adata, obs_key):
    if obs_key is not None:
        obs_data = adata.obs[obs_key]
        unique_cats = obs_data.unique()
        n_cats = len(unique_cats)

        palette = sns.color_palette(
            "tab20" if n_cats > 10 else "tab10", len(unique_cats)
        )
        color_mapping = {category: palette[i] for i, category in enumerate(unique_cats)}

        colors = [color_mapping[e] for e in obs_data.values]
    else:
        colors = "blue"
    return colors


def animate_trajectory(
    positions: np.ndarray,
    interpolation_steps: int = 30,
    out_path: str = "figures/positions.mp4",
    fps: int = 30,
    colors: np.ndarray = None,
):
    """
    positions           (T, N, 2) array from your rollout
    interpolation_steps frames to insert between consecutive rollout steps
    out_path            write-to path for the GIF
    fps                 playback speed
    """
    T, N, _ = positions.shape
    sns.set_theme(style="ticks", context="paper")

    # -------- linear interpolation --------
    frames = []
    for t in range(T - 1):
        start, end = positions[t], positions[t + 1]
        for alpha in np.linspace(0, 1, interpolation_steps, endpoint=False):
            frames.append((1 - alpha) * start + alpha * end)
    frames.append(positions[-1])
    frames = np.asarray(frames)

    # -------- animation --------
    if colors is None:
        colors = cm.nipy_spectral(np.linspace(0, 1, N))

    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    scat = ax.scatter(  # CHANGED ↓
        frames[0][:, 0],
        frames[0][:, 1],  #  start with first frame
        c=colors,
        s=5,
    )
    ax.set_xlim(positions[..., 0].min(), positions[..., 0].max())
    ax.set_ylim(positions[..., 1].min(), positions[..., 1].max())
    ax.set_aspect("equal")

    def init():
        scat.set_offsets(frames[0])
        return (scat,)

    def update(f):
        scat.set_offsets(f)
        return (scat,)

    anim = FuncAnimation(
        fig, update, frames=frames, init_func=init, blit=True, interval=1000 / fps
    )
    # anim.save(out_path, writer=PillowWriter(fps=fps))
    anim.save(out_path, writer=FFMpegWriter(fps=fps))
    plt.close(fig)
