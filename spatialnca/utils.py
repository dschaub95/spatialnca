import numpy as np
import scipy
import networkx as nx
import torch_geometric as pyg
import torch
import matplotlib.pyplot as plt
import random


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
    data = pyg.utils.from_networkx(G)
    data = data.to(device)
    return data


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
