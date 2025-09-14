"""Acquisition function module."""

import gc
import warnings

import torch
from alphaswarm.utils.tensor_types import Float
from alphaswarm.utils.utils import create_lookup_dict
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.objective import GenericMCObjective
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.optim import optimize_acqf_discrete
from botorch.sampling.base import MCSampler
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.sampling import sample_simplex
from rich.progress import MofNCompleteColumn, Progress, SpinnerColumn, TimeElapsedColumn

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

warnings.filterwarnings("ignore")


def optimise_acqf_and_get_suggestion(
    model: ModelListGP,
    acqf: str,
    train_X: Float["n d"],
    train_Y_std: Float["n n_obj"],
    X_norm: Float["s d"],
    sampler: MCSampler,
    batch_size: int,
    max_batch_eval=256,  # default for botorch
) -> Float["{batch_size} m"]:
    """Optimise the acquisition function and return the next suggested point(s).

    Args:
        model: A Botorch Gaussian Process model fitted on training data.
        acqf: The acquisition function strategy to use (qNParEgo or qNEHVI).
        train_X: Training features used to train the model.
        train_Y_std : A tensor of standardised training output.
        X_norm: Normalised search space.
        sampler: A BoTorch MCSampler used to approximate the posterior.
        batch_size: Number of candidates to generate.
        max_batch_eval: Batch size to compute the acquisition function with.

    Returns:
        Batch of points selected by the acquisition function.
    """
    # empty tensor to store new candidates
    X_next = torch.empty((0, X_norm.size(1)), **tkwargs)
    evaluation_matrix = X_norm.detach().clone()

    if acqf == "qNParEgo":
        with torch.no_grad():
            pred = model.posterior(train_X).mean
    else:
        pred = None

    progress = Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        transient=True,
    )

    with progress:
        task = progress.add_task("Optimising batch", total=batch_size)
        for _ in range(batch_size):
            candidate, best_index = _get_next_candidate_and_index(
                acqf=acqf,
                pred=pred,
                model=model,
                train_X=train_X,
                train_Y_std=train_Y_std,
                sampler=sampler,
                X_next=X_next,
                evaluation_matrix=evaluation_matrix,
                max_batch_eval=max_batch_eval,
            )

            X_next = torch.cat((X_next, candidate), dim=0)

            # remove best candidate from the current evaluation matrix
            evaluation_matrix = torch.cat(
                [evaluation_matrix[:best_index], evaluation_matrix[best_index + 1 :]]
            )

            gc.collect()
            torch.cuda.empty_cache()
            progress.advance(task)

    return X_next


def _get_next_candidate_and_index(
    acqf: str,
    pred: torch.Tensor | None,
    model: ModelListGP,
    train_X: torch.Tensor,
    train_Y_std: torch.Tensor,
    sampler: MCSampler,
    X_next: torch.Tensor,
    evaluation_matrix: torch.Tensor,
    max_batch_eval: int,
):
    if acqf == "qNParEgo":
        return setup_qnparego_and_optimize(
            pred,
            model,
            train_X,
            train_Y_std,
            sampler,
            evaluation_matrix,
            max_batch_eval,
        )
    elif acqf == "qNEHVI":
        return setup_qnehvi_and_optimise(
            model,
            sampler,
            train_X,
            train_Y_std,
            X_next,
            evaluation_matrix,
            max_batch_eval,
        )
    else:
        raise ValueError(f"Unsupported acquisition function type: {acqf}")


def setup_qnehvi_and_optimise(
    model: ModelListGP,
    sampler: MCSampler,
    train_X: Float["n d"],
    train_Y_std: Float["n n_obj"],
    X_next: Float["s d"],
    evaluation_matrix: Float["m d"],
    max_batch_eval: int,
) -> torch.Tensor:
    """Set up and optimise the qNEHVI acquisition function.

    This function initialises the acquisition function using the provided
    model and samples, and performs optimisation over a given evaluation
    matrix to find the best candidate point.

    Args:
        model: A Botorch Gaussian Process model fitted on training data.
        sampler: A BoTorch MCSampler used to approximate the posterior
        train_X: Training input data that has already been used to train the model.
        train_Y_std: Standardized training output data corresponding to `train_X`.
        X_next: Evaluation points that are considered in the acquisition function.
        evaluation_matrix: All possible points for evaluating the acquisition.
        max_batch_eval: Maximum number of evaluations that the optimiser will process.

    Returns:
        The single best candidate point from the evaluation matrix as
        determined by optimising the acquisition function.
    """
    ref_point, ref_index = torch.min(
        train_Y_std, axis=0
    )  # ref point taken to be current nadir point.

    acq_func = qLogNoisyExpectedHypervolumeImprovement(
        model=model,
        sampler=sampler,
        ref_point=ref_point,
        incremental_nehvi=True,
        prune_baseline=True,
        X_baseline=train_X,
        X_pending=X_next,
    )

    candidate, acq_value = optimize_acqf_discrete(
        acq_function=acq_func,
        choices=evaluation_matrix,
        max_batch_size=max_batch_eval,
        q=1,
    )

    # get index of best candidate from the current evaluation matrix
    best_index = get_index_from_lookup(candidate, evaluation_matrix)
    best_index = best_index[0]  # for when we get a list back

    return candidate, best_index


def setup_qnparego_and_optimize(
    pred: torch.Tensor,
    model: ModelListGP,
    train_X: torch.Tensor,
    train_Y_std: torch.Tensor,
    sampler: MCSampler,
    evaluation_matrix: torch.Tensor,
    max_batch_eval: int,
):
    """Set up and optimise qNParEgo.

    This function samples weights from a simplex to create a scalarised objective for
    each batch point, then uses these objectives to guide the optimisation process with
    the qLogNoisyExpectedImprovement.

    Args:
        pred: Predicted means from the GP model's posterior over the training data.
        model: A Botorch Gaussian Process model fitted on training data.
        train_X: Tensor containing the training input data.
        train_Y_std: Tensor containing the standardised training output data.
        sampler: A BoTorch MCSampler used to approximate the posterior.
        evaluation_matrix: All possible points for evaluating the acquisition.
        max_batch_eval: Maximum number of evaluations that the optimizer will process.

    Returns:
        The single best candidate point from the evaluation matrix as
        determined by optimising the acquisition function.
    """
    # sample weights uniformly from a d-simplex to create a scalarised objective which
    # is different for every batch point
    weights = sample_simplex(d=train_Y_std.size(1), **tkwargs).squeeze()
    objective = GenericMCObjective(get_chebyshev_scalarization(weights=weights, Y=pred))

    acq_func = qLogNoisyExpectedImprovement(
        model=model,
        objective=objective,
        X_baseline=train_X,
        sampler=sampler,
        prune_baseline=True,
    )

    # Evaluate acquisition function over the discrete search matrix
    candidate, acq_value = optimize_acqf_discrete(
        acq_function=acq_func,
        choices=evaluation_matrix,
        max_batch_size=max_batch_eval,
        q=1,
    )

    # get index of best candidate from the current evaluation matrix
    best_index = get_index_from_lookup(candidate, evaluation_matrix)
    best_index = best_index[0]  # for when we get a list back

    return candidate, best_index


def get_index_from_lookup(X: Float["n m"], X_lookup: Float["n m"]) -> list[int]:
    """Finds the index (or indices) of elements in `X` within `X_lookup`.

    Args:
        X: A tensor containing elements to look up.
        X_lookup: A tensor serving as the lookup source.

    Returns:
        Index of where X is found in X_lookup.

    This function facilitates finding the position(s) of tensor(s) in a lookup tensor,
    useful for indexing or cross-referencing operations.
    """
    # Creates a lookup dictionary to match tensors in X_lookup to their indices
    lookup_dict = create_lookup_dict(X_lookup)

    if X.ndim == 1:
        X = X.unsqueeze(0)

    index_list = []
    for x in X:
        key = tuple(x.tolist())
        if key in lookup_dict:
            index_list.append(lookup_dict[key])

    return index_list
