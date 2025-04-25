import numpy as np
import torch
from tqdm.auto import tqdm
from scipy.spatial import KDTree
from typing import Optional


def uniform_point_cloud(num_points, radius):
    # Generate random angles uniformly distributed between 0 and 2π
    angles = np.random.uniform(0, 2 * np.pi, num_points)

    # Generate radii with uniform distribution within the circle
    radii = np.sqrt(np.random.uniform(0, radius**2, num_points))

    # Convert polar coordinates to Cartesian coordinates
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)

    # Combine into a point cloud
    point_cloud = np.vstack((x, y)).T.astype(np.float32)
    return point_cloud


# random walk transformation
def random_walk(
    points,
    num_steps,
    max_displacement,
    min_displacement=0,
    radius=None,
    xlim=None,
    ylim=None,
    bounds: Optional[list[tuple[float, float]]] = None,
    seed=42,
    device=None,
    progress_bar=True,
    return_intermediate=False,
):
    # handle torch tensors
    if isinstance(points, torch.Tensor):
        device = points.device if device is None else device
        is_torch = True
        points = points.detach().cpu().numpy().astype(np.float32)
    else:
        is_torch = False
        points = np.asarray(points, dtype=float).copy()

    # check the input parameters
    if max_displacement < min_displacement:
        raise ValueError("max_displacement must be >= min_displacement.")

    # check if the points are within the allowed radius or bounds
    if radius is not None and np.any(np.linalg.norm(points, axis=1) > radius):
        raise ValueError("Some points are outside the allowed radius.")
    if xlim is not None and (
        np.any(points[:, 0] < xlim[0]) or np.any(points[:, 0] > xlim[1])
    ):
        raise ValueError("Some points are outside the x-axis bounds.")
    if ylim is not None and (
        np.any(points[:, 1] < ylim[0]) or np.any(points[:, 1] > ylim[1])
    ):
        raise ValueError("Some points are outside the y-axis bounds.")

    if xlim is not None or ylim is not None:
        assert bounds is None, "bounds and xlim/ylim cannot both be provided"
        assert points.shape[1] == 2, "xlim and ylim are only supported for 2D points"
        bounds = [xlim, ylim]

    rng = np.random.default_rng(seed)

    history = []

    for _ in tqdm(range(num_steps), disable=not progress_bar):
        # isotropic directions
        noise_direction = rng.normal(0, 1, size=(len(points), points.shape[1]))
        noise_direction = noise_direction / np.linalg.norm(
            noise_direction, axis=1, keepdims=True
        )
        noise_length = rng.uniform(min_displacement, max_displacement, size=len(points))
        noise_vector = noise_length[:, np.newaxis] * noise_direction
        points += noise_vector

        if radius is not None:
            # Optional: project points back into the radius
            norms = np.linalg.norm(points, axis=1)
            mask = norms > radius
            points[mask] *= radius / norms[mask, np.newaxis]

        if bounds is not None:
            for i, lim in enumerate(bounds):
                if lim is not None:
                    points[:, i] = np.clip(points[:, i], lim[0], lim[1])

        if return_intermediate:
            history.append(
                torch.from_numpy(points).to(device) if is_torch else points.copy()
            )

    if not return_intermediate:
        return torch.from_numpy(points).to(device) if is_torch else points
    else:
        return history


def sunflower_points(n, radius=1.0, median_dist=None, permute=False):
    if median_dist is not None:
        # factor determined empirically
        # d = 1.787 * (r / sqrt(n))
        radius = (median_dist / 1.787) * np.sqrt(n)

    golden_angle = np.pi * (3 - np.sqrt(5))
    i = np.arange(n)
    r = radius * np.sqrt(i / n)
    theta = i * golden_angle
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    coords = np.column_stack((x, y))
    if permute:
        coords = np.random.permutation(coords)
    return coords


def calc_neighbor_dists(points, k_neighbors):
    tree = KDTree(points)
    # Query k-nearest neighbors (excluding self)
    dists, _ = tree.query(points, k=k_neighbors + 1)  # includes self
    neighbor_dists = dists[:, 1:]  # exclude self (distance 0)
    return neighbor_dists
