"""Swarm and particle structures for the PSO algorithm."""

from __future__ import annotations

import logging

import torch

from alphaswarm.utils.tensor_types import Float, typecheck
from alphaswarm.utils.utils import (
    check_tensor,
    get_closest_unknown_features,
    halton,
    latin_hypercube,
    random_initialisation,
    sobol_sequence,
)

init_methods = {
    "random": random_initialisation,
    "sobol": sobol_sequence,
    "lhs": latin_hypercube,
    "halton": halton,
}

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}


class Particle:
    """Particle of the swarm."""

    @typecheck
    def __init__(
        self,
        dim: int,
        position: Float["{dim}"],
        velocity: Float["{dim}"],
        pbest: Float["{dim}"],
        pbest_score: float = 0,
    ) -> None:
        """Particle of the swarm."""
        self.dim = dim  # dimension of the particle's position/velocity
        self.position = position  # position
        self.velocity = velocity  # velocity
        self.pbest = pbest  # personal best position
        self.pbest_score = pbest_score  # personal best score

    def __repr__(self) -> str:
        """Return the string representation of the particle."""
        return (
            f"Particle(dim={self.dim}, position={self.position},"
            + f"velocity={self.velocity}, pbest={self.pbest})"
        )

    @property
    def position(self) -> Float["{self.dim}"]:
        """Return the position of the particle."""
        return self._position

    @position.setter
    def position(self, value: Float["{self.dim}"]) -> None:
        """Set the position of the particle."""
        check_tensor(value, input_shape=(1, self.dim), input_dim=1, name="position")
        self._position = value

    @property
    def velocity(self) -> Float["{self.dim}"]:
        """Return the velocity of the particle."""
        return self._velocity

    @velocity.setter
    def velocity(self, value: Float["{self.dim}"]) -> None:
        """Set the velocity of the particle."""
        check_tensor(value, input_shape=(1, self.dim), input_dim=1, name="velocity")
        self._velocity = value


type Particles = list[Particle]
type IdxExplored = list[list[int]]


class Swarm:
    """Swarm of particles."""

    @typecheck
    def __init__(
        self,
        dim: int,
        n_particles: int,
        init_method: str,
        X_norm: Float["n {dim}"],
        verbose: bool = False,
    ) -> None:
        """Swarm of particles."""
        self.logger = logging.getLogger(__name__)
        self.dim = dim
        self.n_particles = n_particles
        self.init_method = init_method
        self.X_norm = X_norm.to(**tkwargs)
        self.verbose = verbose

        self.particles: Particles = []
        self.log_particles: list[Particles] = []  # log particles
        self.scores: list[Float["{self.n_particles}"]] = []  # scores for PSO
        self.raw_scores: list[Float["{self.n_particles} #obj"]] = []  # raw scores
        self.idx_explored: IdxExplored = []

        # swarm properties
        self.global_best = 0
        self.global_best_position = torch.empty(self.dim)

        if not self.verbose:
            self.logger.disabled = True

    def init_particles(self) -> None:
        """Initialise particles, position and velocity."""
        self.logger.debug(
            f"Initialising particles position with {self.init_method} method"
        )
        init_grid = init_methods[self.init_method.lower()](self.dim, self.n_particles)

        for k in range(self.n_particles):
            position = init_grid[k]
            velocity = torch.zeros(self.dim).to(**tkwargs)
            # NOTE: velocity initialisation can change depending on the algorithm

            self.particles.append(
                Particle(self.dim, position=position, velocity=velocity, pbest=position)
            )

        self._init_clamp_positions()

    def _init_clamp_positions(self) -> None:
        """Clamp the particles' positions to the feature space."""
        self.logger.debug("Clamping positions to the feature space")
        known_indices = set()

        for particle in self.particles:
            particle.position = get_closest_unknown_features(
                particle.position.unsqueeze(0).to(**tkwargs), self.X_norm, known_indices
            ).squeeze(0)
            known_indices.update(
                set(
                    [
                        (self.X_norm == particle.position)
                        .all(axis=1)
                        .nonzero(as_tuple=False)
                        .item()
                    ]
                )
            )
            particle.pbest = particle.position
        self.update_idx_explored()

    @property
    def positions(self) -> Float["{self.n_particles} {self.dim}"]:
        """Return the positions of the particles."""
        return torch.stack([particle.position for particle in self.particles])

    def update_idx_explored(self) -> None:
        """Update indices explored by the swarm."""
        swarm_positions = self.positions
        indices = (
            (self.X_norm == swarm_positions.unsqueeze(1))
            .all(dim=2)
            .nonzero(as_tuple=False)[:, 1]
        )
        self.idx_explored.append(indices.tolist())

    def __len__(self) -> int:
        """Return the number of particles in the swarm."""
        return len(self.particles)

    def __repr__(self) -> str:
        """Return the string representation of the swarm."""
        return (
            f"Swarm(dim={self.dim}, n_particles={self.n_particles},"
            + f"init_method={self.init_method})"
        )
