import numpy as np
import scipy
import networkx as nx
import torch_geometric as pyg
import torch
import matplotlib.pyplot as plt
import random
import seaborn as sns
import scanpy as sc


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
    adata = sc.AnnData(X=np.zeros((coords.shape[0], 100)), obsm={"spatial": coords})
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


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)


def spatial_scatter(adata, color=None, pos_key="spatial", cmap=None):
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
    plt.show()
