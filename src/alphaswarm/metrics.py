"""Metrics to compute for a given optimisation."""

import numpy as np
import torch
from torch import Tensor

from alphaswarm.objective_functions import MultiObjective
from alphaswarm.swarms import Swarm
from alphaswarm.utils.moo_utils import (
    diversity,
    get_pareto_points,
    hypervolume,
    igd,
    igd_plus,
)
from alphaswarm.utils.tensor_types import Float, typecheck
from alphaswarm.utils.utils import nadir_point

type Scores = list[float]


class Metrics:
    """Metrics of the optimisation process."""

    @typecheck
    def __init__(
        self, swarm: Swarm, X_norm: Float["n m"], y_raw_target: Float["n q"]
    ) -> None:
        """Initialisation of the metrcis from a final swarm."""
        self.swarm = swarm
        self.X_norm = X_norm
        self.y_raw_target = y_raw_target
        self.objective_function = MultiObjective(X_norm, y_raw_target)
        self.nadir_pt = nadir_point(y_raw_target)
        self.tot_hypervolume = hypervolume(Y=y_raw_target, ref_point=self.nadir_pt)
        self.explored_y = [
            torch.cat([Tensor(scores) for scores in self.swarm.raw_scores[: k + 1]])
            for k in range(len(self.swarm.raw_scores))
        ]
        self.explored_indices = [
            torch.cat(
                [Tensor(idx) for idx in self.swarm.idx_explored[: k + 1]]
            ).tolist()
            for k in range(len(self.swarm.idx_explored))
        ]

    def get_scores(self) -> list[Scores]:
        """Get the scores of the swarm."""
        return self.swarm.scores

    def get_explored_indices(self) -> list[list[int]]:
        """Get the indices of the explored solutions."""
        return self.swarm.idx_explored

    def get_score_of_iter(self, iteration: int) -> Scores:
        """Get the scores of the swarm at a given iteration."""
        assert iteration < len(
            self.swarm.scores
        ), f"Iteration should be less than {len(self.swarm.scores)}"
        return self.swarm.scores[iteration]

    def get_hypervolume(self) -> Scores:
        """Get the hypervolume of the swarm at each iteration."""
        return [
            (
                hypervolume(Y=self.explored_y[i], ref_point=self.nadir_pt)
                / self.tot_hypervolume
            )
            for i in range(len(self.swarm.raw_scores))
        ]

    def get_hypervolume_of_iter(self, iteration: int) -> float:
        """Compute the hypervolume of the swarm at a given iteration."""
        assert iteration < len(
            self.swarm.raw_scores
        ), f"iter should be less than {len(self.swarm.raw_scores)}"
        return self.get_hypervolume()[iteration]

    def get_log_hypervolume_regret(self) -> Scores:
        """Compute the hypervolume regret."""
        hypervolume_diff = [
            (
                self.tot_hypervolume
                - hypervolume(Y=self.explored_y[i], ref_point=self.nadir_pt)
            )
            for i in range(len(self.swarm.raw_scores))
        ]
        return [np.log(i) if i != 0 else 0 for i in hypervolume_diff]

    def get_igd(self):
        """Compute the IGD metric."""
        pareto_true = get_pareto_points(self.y_raw_target)
        pareto_fronts = [get_pareto_points(i) for i in self.explored_y]
        return [igd(pareto_true, k) for k in pareto_fronts]

    def get_igd_plus(self):
        """Compute the IGD+ metric."""
        pareto_true = get_pareto_points(self.y_raw_target)
        pareto_fronts = [get_pareto_points(i) for i in self.explored_y]
        return [igd_plus(pareto_true, k) for k in pareto_fronts]

    def get_top_n(self, n: int, utopia_point: list[int]) -> list[float]:
        """Compute the top-n% metric."""
        # Compute the ranking of the solutions based on the distance to the utopia point
        distances = torch.cdist(self.y_raw_target, Tensor(utopia_point).unsqueeze(0))
        ranking = torch.argsort(distances, dim=0).squeeze()
        top_n = ranking[:n]

        top_n_iter = []
        for iteration_idx in self.explored_indices:
            # Compute the size of the intersection between the top n solutions and the
            # explored solutions
            top_n_explored = len(set(top_n.tolist()).intersection(set(iteration_idx)))
            top_n_iter.append(top_n_explored / n)
        return top_n_iter

    def get_pairwise_distance(self) -> list[Float["n n"]]:
        """Return the distance matrix between each sampled points for each iteration."""
        pairwise_distances = []

        for batch in self.swarm.idx_explored:
            X_sampled = self.X_norm[batch]
            pairwise_distance = torch.cdist(X_sampled, X_sampled)
            pairwise_distances.append(pairwise_distance)
        return pairwise_distances

    def get_diversity(self):
        """Return the diversity of each batch based on the pairwise distance."""
        diversity_scores = []

        for batch in self.swarm.idx_explored:
            X_sampled = self.X_norm[batch]
            diversity_score = diversity(X_sampled)
            diversity_scores.append(diversity_score)
        return diversity_scores
