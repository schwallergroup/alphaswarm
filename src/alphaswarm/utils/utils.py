"""Utils functions for the PSO algorithm."""

import torch
from alphaswarm.utils.tensor_types import Float, typecheck
from scipy.stats.qmc import Halton, LatinHypercube
from sklearn.preprocessing import MinMaxScaler
from torch import Tensor
from torch.quasirandom import SobolEngine


@typecheck
def normalise_features[T: Float["n m"]](X_features: T) -> T:
    """Normalise the feature space."""
    scaler = MinMaxScaler()
    scaler.fit(X_features)
    return Tensor(scaler.transform(X_features))


@typecheck
def get_closest_features[T: Float["n m"]](x: T, feature_space: T) -> T:
    """Get the closest feature in the grid."""
    indices = torch.cdist(x, feature_space).argmin(dim=1)
    return feature_space[indices]


@typecheck
def get_closest_unknown_features[T: Float["n m"]](
    x: T, feature_space: T, known_indices: set[int]
) -> T:
    """Get the closest feature in the grid from the unknown features."""
    distances = torch.cdist(x, feature_space)
    closest_idx = torch.argsort(distances, axis=1)
    correct_idx = []
    for vec in closest_idx:
        for idx in vec:
            if idx.item() not in known_indices:
                correct_idx.append(idx.item())
                break

    return feature_space[correct_idx]


def get_unique_indices(indices: list[list[int]]) -> set[int]:
    """Get the unique indices from a list of indices."""
    return set([item for sublist in indices for item in sublist])


@typecheck
def update_known_indices(
    known_indices: set, position: Float[""], X_norm: Float["n m"]
) -> None:
    """Update the set of known indices with the index of the given position.

    Args:
        known_indices: Set of indices already explored.
        position: The position tensor to find in X_norm.
        X_norm: The normalized swarm positions.
    """
    index = (X_norm == position).all(axis=1).nonzero(as_tuple=False).item()
    known_indices.add(index)


@typecheck
def check_tensor(
    tensor: Float["n m"], input_shape: tuple[int, int | None], input_dim: int, name: str
) -> None:
    """Check if the Tensor is a 1D Tensor with the correct dimensions."""
    shape = tensor.shape
    if shape[0] != input_shape[1]:
        raise ValueError(f"{name.title()} must have a length of {input_dim}.")
    if tensor.ndim != input_dim:
        raise ValueError(f"{name.title()} must be a 1D Tensor.")


def random_initialisation(dim: int, n_particles: int) -> Float["{n_particles} {dim}"]:
    """Random initialisation of particle positions in a unit hypercube."""
    return torch.rand((n_particles, dim))


def sobol_sequence(dim: int, n_particles: int) -> Float["{n_particles} {dim}"]:
    """Sobol initialisation of particle positions in a unit hypercube."""
    sobol_sampler = SobolEngine(dimension=dim, scramble=True)
    sobol_grid = sobol_sampler.draw(n_particles).double()  # convert to float64
    return sobol_grid


def latin_hypercube(dim: int, n_particles: int) -> Float["{n_particles} {dim}"]:
    """Latin Hypercube initialisation of particle positions in a unit hypercube."""
    sampler = LatinHypercube(d=dim, scramble=True)
    return Tensor(sampler.random(n=n_particles)).double()


def halton(dim: int, n_particles: int) -> Float["{n_particles} {dim}"]:
    """Halton initialisation of particle positions in a unit hypercube."""
    sampler = Halton(d=dim, scramble=True)
    return Tensor(sampler.random(n_particles)).double()


@typecheck
def nadir_point(Y: Float["n n_obj"]) -> Float["n_obj"]:
    """Compute the nadir point."""
    return Y.min(dim=0).values


@typecheck
def create_lookup_dict(X_lookup: Float["n m"]) -> dict:
    """Create a lookup dictionary from an iterable of tensors to their indices.

    Args:
        X_lookup: An iterable tensor where each row is used as lookup values.

    Returns:
        dict: A dictionary mapping tensor tuples to their respective indices.
    """
    return {tuple(x.tolist()): i for i, x in enumerate(X_lookup)}
