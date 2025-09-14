"""Utility functions for multi-objective optimisation."""

from math import log

import torch
from alphaswarm.utils.tensor_types import Float, typecheck
from botorch.utils.multi_objective import is_non_dominated
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from torch import Tensor


@typecheck
def hypervolume(Y: Float["n n_obj"], *, ref_point: Float["n_obj"]) -> float:
    """Compute the hypervolume HV(P, r) of a set of points from a reference.

    The input points must be unstandardised (raw values from the objective
    functions).

    Args:
        Y: `n` points, each with `n_obj` dimensions. Each dimension represents an
            objective to maximise in the optimisation problem.
        ref_point: The reference point. Usually the nadir point.

    Returns:
        The hypervolume based on the computed non-dominated partitioning.
    """
    bd = NondominatedPartitioning(ref_point=ref_point, Y=Y)
    return bd.compute_hypervolume().item()


@typecheck
def hypervolume_improvement(
    Y: Float["n n_obj"], *, Y_new: Float["#new n_obj"], ref_point: Float["n_obj"]
) -> float:
    """Compute the hypervolume improvement HVI(P, Q, r) = HV(P ∪ Q, r) - HV(P, r).

    Args:
        Y: `n` points, each with `n_obj` dimensions.
        Y_new: The new set of points / a new point with `n_obj` dimensions.
        ref_point: The reference point. Usually the nadir point.

    Returns:
        The hypervolume improvement based on the computed non-dominated partitioning.
    """
    union_Y = torch.cat([Y, Y_new], dim=0)
    return hypervolume(union_Y, ref_point=ref_point) - hypervolume(
        Y, ref_point=ref_point
    )


def hypervolume_regret[F: float](achieved_hypervolume: F, optimal_hypervolume: F) -> F:
    """Computes the hypervolume regret.

    HVR = V_optimal - V_achieved

    Args:
        achieved_hypervolume: The hypervolume achieved by the optimisation process.
        optimal_hypervolume: The maximum possible or reference hypervolume.

    Returns:
        The hypervolume regret.
    """
    return optimal_hypervolume - achieved_hypervolume


def log_hypervolume_regret[F: float](
    achieved_hypervolume: F, optimal_hypervolume: F
) -> F:
    """Computes the logarithm of the hypervolume regret.

    lHVR = ln(V_optimal - V_achieved) if V_optimal > V_achieved, otherwise 0.

    Args:
        achieved_hypervolume: The hypervolume achieved by the optimisation process.
        optimal_hypervolume: The maximum possible or reference hypervolume.

    Returns:
        The natural logarithm of the hypervolume regret if positive, otherwise 0.
    """
    hvr = hypervolume_regret(achieved_hypervolume, optimal_hypervolume)
    return log(hvr) if hvr > 0 else 0


@typecheck
def get_pareto_points(Y: Float["n n_obj"]) -> Float["m n_obj"]:
    """Identify and return the Pareto-optimal points from a given set.

    Args:
        Y : A tensor of `n` points, each with `n_obj` dimensions. Each dimension
            represents an objective to maximise in the optimisation problem.

    Returns:
        A tensor containing only the Pareto-optimal points from the input set.
    """
    return Y[is_non_dominated(Y)]


@typecheck
def igd(pf_true: Float["n n_obj"], pf_approx: Float["m n_obj"]) -> float:
    """Calculate the Inverted Generational Distance (IGD).

    IGD measures the average Euclidean distance from each point in the true
    Pareto front to the nearest point in the approximated front.

    Args:
        pf_true: `n` Pareto-optimal points, each with `n_obj` dimensions.
        pf_approx: `m` Pareto approximated points, each with `n_obj` dimensions.

    Returns:
        The calculated IGD.
    """
    distances = Tensor(
        [torch.min(torch.norm(pf_true - approx_point)) for approx_point in pf_approx]
    )
    igd = distances.mean().item()

    return igd


@typecheck
def igd_plus_distance_maximisation(
    true_point: Float["n_obj"], approx_point: Float["n_obj"]
) -> float:
    """Calculate the IGD+ distance for maximisation problems.

    ref: https://doi.org/10.1007/978-3-030-12598-1_27

    Args:
        true_point: The reference point `z` taken from the true PF.
        approx_point: The solution point `a` taken from the approximate/solution PF.

    Returns:
        The IGD+ distance.
    """
    distances = torch.maximum(true_point - approx_point, torch.zeros_like(true_point))

    # Calculate the Euclidean distance
    return torch.sqrt(torch.sum(distances**2)).item()


@typecheck
def igd_plus(pf_true: Float["n n_obj"], pf_approx: Float["m n_obj"]) -> float:
    """Calculate the Inverted Generational Distance Plus (IGD+).

    ref: https://doi.org/10.1007/978-3-030-12598-1_27

    Args:
        pf_true: True Pareto front for `n_obj` objectives with `n` points.
        pf_approx: Approximated Pareto front for `n_obj` objectives with `m` points.

    Returns:
        The calculated IGD+.
    """
    distances = [
        min(
            igd_plus_distance_maximisation(true_point, approx_point)
            for approx_point in pf_approx
        )
        for true_point in pf_true
    ]
    return Tensor(distances).mean().item()


@typecheck
def diversity(X_sampled: Float["n dim"]) -> float:
    """Compute the diversity of a set of points."""
    pairwise_distances = torch.cdist(X_sampled, X_sampled)
    return pairwise_distances[
        torch.tril(torch.ones_like(pairwise_distances), diagonal=-1).bool()
    ].tolist()
