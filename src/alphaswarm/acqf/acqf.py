"""Acquisition functions for the optimisation algorithm."""

import inspect
from collections.abc import Callable
from typing import Any, NamedTuple

import torch
from alphaswarm.acqf.acqfunc import optimise_acqf_and_get_suggestion
from alphaswarm.models.gp import get_models_pred
from alphaswarm.swarms import Particle, Swarm
from alphaswarm.utils.tensor_types import Float, typecheck
from alphaswarm.utils.utils import (
    get_closest_unknown_features,
    get_unique_indices,
    sobol_sequence,
    update_known_indices,
)
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.transforms import standardize
from sklearn.preprocessing import StandardScaler
from torch import Tensor

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}


def acqf_manager(func: Callable, **kwargs) -> Callable:
    """Decorator to manage the acquisition functions."""
    valid_kwargs = inspect.signature(func).parameters

    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_kwargs}

    return func(**filtered_kwargs)


def update_particle_position(
    particle: Particle,
    new_position: Float["n_particles n_features"],
    known_indices: set[int],
    swarm_X_norm: Float["n m"],
) -> None:
    """Update the position of a particle.

    Args:
        particle: Particle object.
        new_position: New position of the particle.
        known_indices: Indices of the known points.
        swarm_X_norm: Normalised features of the swarm.
    """
    particle.position = get_closest_unknown_features(
        new_position.unsqueeze(0), swarm_X_norm, known_indices
    ).squeeze(0)
    update_known_indices(known_indices, particle.position, swarm_X_norm)


@typecheck
def alpha_pso(
    swarm: Swarm,
    pso_params: NamedTuple,
    gps: ModelListGP,
    X_train: Float["n m"],
    y_train: Float["n #obj"],
    seed: int,
    iteration: int,
) -> Swarm:
    """PSO algorithm guided by the qNEHVI acquisition function.

    Args:
        swarm: Current swarm of particles.
        pso_params: Parameters for the PSO algorithm.
        gps: Gaussian Process models.
        X_train: Training features of all data points.
        y_train: Training target values of all data points.
        seed: Random seed.
        iteration: Current iteration number.

    Returns:
        Swarm of particles with the new positions and velocities updated.
    """
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([len(swarm)]), seed=seed)
    # Get the ordered acqf positions
    acqf_positions = optimise_acqf_and_get_suggestion(
        model=gps,
        acqf="qNEHVI",
        train_X=X_train,
        train_Y_std=standardize(y_train),
        X_norm=swarm.X_norm.to(**tkwargs),
        sampler=sampler,
        batch_size=len(swarm),
    )

    particles_to_move = pso_params.n_particles_to_move[iteration - 1]

    # Get the indices of the worst particle
    worst_particles_indices = swarm.scores[-1].argsort()[:particles_to_move]

    known_indices = get_unique_indices(swarm.idx_explored)
    acqf_indices = list(range(particles_to_move))
    # Remove the particles to move from the acqf positions
    acqf_guide = acqf_positions[particles_to_move:]

    for i, particle in enumerate(swarm.particles):
        if i in worst_particles_indices:
            # Assign the acqf position to the worst particles
            new_position = get_closest_unknown_features(
                acqf_positions[acqf_indices[0]].unsqueeze(0),
                swarm.X_norm,
                known_indices,
            ).squeeze(0)
            particle.position = new_position
            particle.velocity = torch.normal(mean=0, std=1, size=(swarm.dim,)).to(
                **tkwargs
            )
            acqf_indices.pop(0)

        else:
            # Generate random vectors
            phi_1 = pso_params.c_1 * torch.rand(swarm.dim).to(**tkwargs)
            phi_2 = pso_params.c_2 * torch.rand(swarm.dim).to(**tkwargs)
            phi_a = pso_params.c_a * torch.rand(swarm.dim).to(**tkwargs)

            # Update the velocity
            particle.velocity = (
                pso_params.w * particle.velocity
                + (particle.pbest - particle.position) * phi_1
                + (swarm.global_best_position - particle.position) * phi_2
                + (acqf_guide[0] - particle.position) * phi_a
            )

            acqf_guide = acqf_guide[1:]  # Remove the first acqf position

            # Update the position
            new_position = particle.position + particle.velocity
            # Clamp the position to the grid
            update_particle_position(
                particle, new_position, known_indices, swarm.X_norm
            )

    return swarm


@typecheck
def qNEHVI(
    swarm: Swarm,
    gps: ModelListGP,
    X_train: Float["n_obs n_features"],
    y_train: Float["n_obs n_obj"],
    seed: int,
) -> Swarm:
    """Move the Swarm directly to the positions suggested by the qNEHVI function.

    Args:
        swarm: Swarm object.
        gps: Gaussian Process models.
        X_train: Training features.
        y_train: Training target values.
        seed: Random seed.

    Returns:
        Swarm of particles with the new positions and velocities updated.
    """
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([len(swarm)]), seed=seed)
    acqf_positions = optimise_acqf_and_get_suggestion(
        model=gps,
        acqf="qNEHVI",
        train_X=X_train,
        train_Y_std=standardize(y_train),
        X_norm=swarm.X_norm.to(**tkwargs),
        sampler=sampler,
        batch_size=len(swarm),
    )
    known_indices = get_unique_indices(swarm.idx_explored)

    # Move all particles to the acqf positions
    for i, particle in enumerate(swarm.particles):
        new_position = acqf_positions[i]
        # Clamp the position to the grid
        update_particle_position(particle, new_position, known_indices, swarm.X_norm)
        # Set the velocity to zero
        particle.velocity = torch.zeros(swarm.dim).to(**tkwargs)

    return swarm


@typecheck
def exploitative_BO(
    swarm: Swarm,
    gps: ModelListGP,
    y_train: Float["n #obj"],
) -> Swarm:
    """Run an exploitative BO step by moving particles to the highest predicted mean.

    Args:
        swarm: Swarm object.
        gps: Gaussian Process models.
        y_train: Training target values.

    Returns:
        Swarm of particles with the new positions and velocities updated.
    """
    # get the scaler
    scaler = StandardScaler().fit(y_train.detach().cpu())
    # get the models predictions
    y_pred = get_models_pred(gps, swarm.X_norm, scaler).mean
    # define a utopia point
    utopia_point = Tensor([1.1, 1.1])
    # calculate the distance to the utopia point
    utopia_distances = torch.norm(y_pred - utopia_point, dim=1)
    # get the indices of the particles with the highest predicted mean
    best_indices = utopia_distances.argsort()[: len(swarm)]
    # get the positions of the particles with the highest predicted mean
    best_positions = swarm.X_norm[best_indices]

    known_indices = get_unique_indices(swarm.idx_explored)

    for i, particle in enumerate(swarm.particles):
        new_position = best_positions[i]
        update_particle_position(particle, new_position, known_indices, swarm.X_norm)
        particle.velocity = torch.zeros(swarm.dim).to(**tkwargs)

    return swarm


def sobol(swarm: Swarm) -> Swarm:
    """Quasi-random search with Sobol sequence.

    Args:
        swarm: Swarm object

    Returns:
        Swarm of particles with the new positions and velocities updated.
    """
    new_positions = sobol_sequence(swarm.dim, len(swarm)).to(**tkwargs)

    known_indices = get_unique_indices(swarm.idx_explored)

    for i, particle in enumerate(swarm.particles):
        new_positions
        update_particle_position(
            particle, new_positions[i], known_indices, swarm.X_norm
        )
        particle.velocity = torch.zeros(swarm.dim).to(**tkwargs)

    return swarm


def canonical_pso(
    swarm: Swarm,
    pso_params: dict[str, Any],
) -> Swarm:
    """Canonical (batch) Particle Swarm Optimisation algorithm.

    This is a modified version of PSO, adapted for batch evaluations.
    The pbest and gbest positions are only updated after all particles have been moved.

    Args:
        swarm: Swarm object
        pso_params: parameters for the swarm algorithm

    Returns:
        Swarm of particles with the new positions and velocities updated.
    """
    known_indices = get_unique_indices(swarm.idx_explored)

    for particle in swarm.particles:
        # Generate random vectors
        phi_1 = pso_params.c_1 * torch.rand(swarm.dim).to(**tkwargs)
        phi_2 = pso_params.c_2 * torch.rand(swarm.dim).to(**tkwargs)

        # Update the velocity
        particle.velocity = (
            pso_params.w * particle.velocity
            + (particle.pbest - particle.position) * phi_1
            + (swarm.global_best_position - particle.position) * phi_2
        )

        # Update the position
        new_position = particle.position + particle.velocity

        update_particle_position(particle, new_position, known_indices, swarm.X_norm)

    return swarm
