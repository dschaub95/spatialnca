import numpy as np
import torch
from scipy.spatial import KDTree


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
def random_walk_2d(
    points,
    num_steps,
    max_displacement,
    min_displacement=0,
    radius=None,
    xlim=None,
    ylim=None,
    seed=42,
    device=None,
):
    # handle torch tensors
    if isinstance(points, torch.Tensor):
        device = points.device if device is None else device
        is_torch = True
        points = points.detach().cpu().numpy().astype(np.float32)
    else:
        is_torch = False
        points = np.asarray(points, dtype=float)

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

    rng = np.random.default_rng(seed)

    current = points.copy()

    for _ in range(num_steps):
        # Start with all points needing updates
        points_to_update = np.arange(len(current))

        while len(points_to_update) > 0:
            angles = rng.uniform(0, 2 * np.pi, size=len(points_to_update))
            disps = rng.uniform(
                min_displacement, max_displacement, size=len(points_to_update)
            )
            moves = disps[:, np.newaxis] * np.column_stack(
                (np.cos(angles), np.sin(angles))
            )
            new_positions = current[points_to_update] + moves

            if radius is not None:
                valid = np.linalg.norm(new_positions, axis=1) <= radius
                # Update the valid points
                current[points_to_update[valid]] = new_positions[valid]
                # Keep trying for the invalid points
                points_to_update = points_to_update[~valid]
            elif xlim is not None or ylim is not None:
                # check if the new positions are within the bounds
                valid = (
                    (new_positions[:, 0] >= xlim[0])
                    & (new_positions[:, 0] <= xlim[1])
                    & (new_positions[:, 1] >= ylim[0])
                    & (new_positions[:, 1] <= ylim[1])
                )
                # Update the valid points
                current[points_to_update[valid]] = new_positions[valid]
                # Keep trying for the invalid points
                points_to_update = points_to_update[~valid]
            else:
                current[points_to_update] = new_positions
                points_to_update = np.empty(0, dtype=int)

    if is_torch:
        return torch.tensor(current, device=device)
    else:
        return current


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
