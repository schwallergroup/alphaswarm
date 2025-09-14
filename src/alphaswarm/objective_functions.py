"""Objective functions and scalarisation methods."""

import torch
from torch import Tensor

from alphaswarm.utils.tensor_types import Float, Int


class SingleObjective:
    """Single objective function."""

    def __init__(self, X_norm: Float["n m"], y_target: Float["n"], noise=0.0):
        """Initialise the objective space.

        It is assumed that the inputs observations are already clamped to the grid.
        """
        self.X_norm = X_norm
        self.y_target = y_target

        if noise > 0.0:
            self.y_target += noise * torch.randn_like(self.y_target)

    def get_indices(self, x: Float["n m"]) -> Int["n"]:
        """Return the indices of the given input."""
        indices = (
            (self.X_norm == x.unsqueeze(1)).all(dim=2).nonzero(as_tuple=False)[:, 1]
        )
        return indices

    def target(self, x: Float["n m"]) -> Float["n"]:
        """Return the target values for the given input."""
        indices = self.get_indices(x)
        return self.y_target[indices]

    def single_objective(self, X_obs: Float["n m"]) -> Float["n"]:
        """Return the target values."""
        return self.target(X_obs).unsqueeze(0), self.target(X_obs).unsqueeze(0)


class MultiObjective:
    """Multi-objective function."""

    def __init__(
        self, X_norm: Float["n m"], y_target: Float["n n_obj"], noise: float = 0.0
    ):
        """Initialise the objective space.

        It is assumed that the inputs observations are already clamped to the grid.

        Args:
            X_norm: The normalised input features.
            y_target: The target values.
            noise: The noise level.
        """
        self.X_norm = X_norm
        self.y_target = y_target

        if noise > 0.0:
            self.y_target += noise * torch.randn_like(self.y_target)

    def get_indices(self, x: Float["n m"]) -> Int["n"]:
        """Return the indices of the given input."""
        indices = (
            (self.X_norm == x.unsqueeze(1)).all(dim=2).nonzero(as_tuple=False)[:, 1]
        )
        return indices

    def target(self, x: Float["n_obs dim"]) -> Float["n_obs n_obj"]:
        """Return the target values for the given input."""
        indices = self.get_indices(x)
        return self.y_target[indices]

    def weighted_sum(
        self, X_obs: Float["n_obs dim"], weights: list[float]
    ) -> tuple[Float["n_obj"], Float["n n_obj"]]:
        """Weighted Sum Multi-objective function."""
        target = self.target(X_obs)
        return torch.sum(Tensor(weights) * target, dim=1), target

    def weighted_power(
        self, X_obs: Float["n_obs dim"], weights: list[float], p: float
    ) -> tuple[Float["n_obj"], Float["n n_obj"]]:
        """Weighted Power Multi-objective function."""
        target = self.target(X_obs)
        return torch.sum(Tensor(weights) * (self.target(X_obs) ** p), dim=1), target

    def weighted_norm(
        self, X_obs: Float["n_obs dim"], weights: list[float], p: float
    ) -> Float["n_obs"]:
        """Weighted Norm (L_p norm)."""
        target = self.target(X_obs)
        return (torch.sum(Tensor(weights) * torch.abs(target) ** p, dim=1)) ** (
            1 / p
        ), target

    def weighted_product(
        self, X_obs: Float["n_obs dim"], weights: list[float]
    ) -> Float["n"]:
        """Weighted Product."""
        target = self.target(X_obs)
        return torch.prod(target ** Tensor(weights), dim=1), target

    def chebyshev_function(
        self,
        X_obs: Float["n_obs dim"],
        weights: list[float],
        utopian_pt: Float["n"],
    ) -> Float["n_obs"]:
        """Chebyshev function with utopian point."""
        target = self.target(X_obs)
        return torch.min(
            Tensor(weights) * torch.abs(target - utopian_pt), dim=1
        ).values, target

    def augmented_chebyshev(
        self,
        X_obs: Float["n_obs dim"],
        weights: list[float],
        utopian_pt: Float["n m"],
        alpha: float,
    ) -> Float["n_obs"]:
        """Augmented Chebyshev function with utopian point."""
        target = self.target(X_obs)
        return torch.min(
            Tensor(weights) * torch.abs(target - utopian_pt), dim=1
        ).values + alpha * torch.sum(torch.abs(target - utopian_pt), dim=1), target

    def modified_chebyshev(
        self,
        X_obs: Float["n_obs dim"],
        weights: list[float],
        utopian_pt: Float["n"],
        alpha: float,
    ) -> Float["n_obs"]:
        """Return modified Chebyshev function from an utopian point."""
        target = self.target(X_obs)
        return torch.min(
            Tensor(weights) * torch.abs(target - utopian_pt)
            + alpha * torch.sum(torch.abs(target - utopian_pt), dim=1),
            dim=1,
        ).values, target
