"""Main module for the Particle Swarm Optimisation algorithm."""

import logging
import os
from collections import namedtuple

import pandas as pd
import pytorch_lightning as pl
import torch
from torch import Tensor

from alphaswarm.acqf.acqf import (
    acqf_manager,
    alpha_pso,
    canonical_pso,
    exploitative_BO,
    qNEHVI,
    sobol,
)
from alphaswarm.configs import PSOBenchmarkConfig, PSOExperimentConfig
from alphaswarm.models import gp
from alphaswarm.objective_functions import MultiObjective, SingleObjective
from alphaswarm.swarms import Swarm
from alphaswarm.utils.tensor_types import Float, typecheck
from alphaswarm.utils.utils import normalise_features

algos_mapping = {
    "canonical-pso": canonical_pso,
    "alpha-pso": alpha_pso,
    "qnehvi": qNEHVI,
    "sobol": sobol,
    "exploitative-bo": exploitative_BO,
}

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}


class PSO:
    """Particle Swarm Optimisation class."""

    @typecheck
    def __init__(self, config: PSOBenchmarkConfig | PSOExperimentConfig) -> None:
        """Initialisation of the optimisation from a config."""
        self.log = logging.getLogger(__name__)  # init logger
        self.config = config
        self.gp_config = self.config.model_config  # GP model configuration

        self.X_features: Float["n m"] = Tensor(config.X_features).double()
        self.X_norm = normalise_features(self.X_features)  # normalise features

        self.dim = self.X_norm.shape[1]  # number of features
        self.algo = algos_mapping[config.algo.lower()]  # algorithm to update the swarm
        self.pso_params = config.pso_params  # parameters for the swarm algorithm

        if isinstance(config, PSOBenchmarkConfig):
            self.log.info("Initialising benchmark experiment")
            self._init_benchmark_params()  # if benchmarking, initialise the params

        # Set random seed
        pl.seed_everything(self.config.seed, verbose=False)
        self.log.info(f"Global random seed set to {self.config.seed}")

        # GPU check
        if torch.cuda.is_available():
            self.log.info("GPU available")
            self.log.info(f"Running on {torch.cuda.get_device_name()}")
        else:
            self.log.warning("Running on CPU, runtime may be slower")

        # Initialise the swarm
        self.swarm = Swarm(
            dim=self.dim,
            n_particles=self.config.n_particles,
            init_method=self.config.init_method,
            X_norm=self.X_norm,
        )
        self.swarm.init_particles()  # initialise particles
        self.log.info(f"Swarm initialised with {config.n_particles} particles")
        self.log.info(f"Initial positions set with {config.init_method}")

    def _init_benchmark_params(self) -> None:
        """Initialise the parameters for the benchmark experiments."""
        self.y_target: Float["n #obj"] = Tensor(
            self.config.y_target
        ).double()  # target values (single or multi-objective)

        self.n_iter = self.config.n_iter  # maximum number of iterations

        self.obj_func = (
            self.config.objective_function
        )  # function to evaluate the objective function
        self.obj_func_params = (
            self.config.obj_func_params
        )  # parameters of the objective function

        # Objective function
        single_obj = SingleObjective(self.X_norm, self.y_target)
        multi_obj = MultiObjective(
            self.X_norm, self.y_target, noise=self.obj_func_params["noise"]
        )
        self.obj_func_params.pop("noise")

        obj_func_mapping = {
            "single": single_obj.single_objective,
            "weighted_sum": multi_obj.weighted_sum,
        }
        self.objective_function = obj_func_mapping[self.config.objective_function]

    def _init_suggest(self) -> None:
        """Initial setup for the suggest method."""
        if not isinstance(self.config, PSOExperimentConfig):
            self.log.error(
                "Suggest method only available for PSOExperimentConfig config."
            )
            return

        if self.config.iteration_number > 1:
            for i in range(1, self.config.iteration_number):
                suggestions_file = os.path.join(
                    self.config.pso_suggestions_path,
                    self.config.pso_suggestions_format.format(i),
                )
                if not os.path.isfile(suggestions_file):
                    self.log.warning(f"Suggestions file {suggestions_file} not found.")

        if not os.path.exists(self.config.experimental_data_path):
            os.makedirs(self.config.experimental_data_path)
            self.log.info(f"Created folder {self.config.experimental_data_path}")

    def suggest(self) -> None:
        """Suggest the next experiments to run)."""
        self._init_suggest()  # inital setup for the suggest method

        self.log.info("Starting the optimisation process")
        self.log.info(f"Chemical space size: {self.X_norm.shape[0]}")
        self.log.info(f"Number of features: {self.X_norm.shape[1]}")
        self.log.info(f"Objectives: {self.config.objective_columns}")
        self.log.info(f"Method: {self.config.algo}")
        self.log.info(f"Number of particles: {self.config.n_particles}")

        # Initialisation step for iteration_number = 1
        if self.config.iteration_number == 1:
            # one-off bootstrap -> done
            self._write_initial_suggestions()
            return

        training_files = self._collect_training_files()
        Xs_train, ys_train = self._load_training_tensors(training_files)

        self._bootstrap_swarm_state(Xs_train[0], ys_train[0])
        gps, X_train, y_train = self._train_initial_gps()

        self._iterative_optimisation(gps, X_train, y_train, Xs_train, ys_train)
        self._write_final_suggestions()

    def _write_initial_suggestions(self) -> None:
        """Write the initial suggestions (iteration 1) to a file."""
        file_path = os.path.join(
            self.config.pso_suggestions_path,
            self.config.pso_suggestions_format.format("1"),
        )
        if os.path.isfile(file_path):
            self.log.error(f"Suggestions file {file_path} already exists.")
            return

        suggestions = self.swarm.idx_explored[0]
        df = self.config.raw_df.iloc[suggestions]
        os.makedirs(self.config.pso_suggestions_path, exist_ok=True)
        df.to_csv(file_path, index=False)
        self.log.info(f"Initial {len(df)} suggestions -> {file_path}")

    # 2) gather & validate training files
    def _collect_training_files(self) -> list[str]:
        """Collect and validate the experimental training files."""
        training_files: list[str] = []
        for i in range(1, self.config.iteration_number):
            path = os.path.join(
                self.config.experimental_data_path,
                self.config.experimental_data_format.format(i),
            )
            if not os.path.exists(path):
                self.log.critical(f"Missing experimental data file {path}.")
                raise FileNotFoundError(path)
            training_files.append(path)
        return training_files

    def _load_training_tensors(
        self, files: list[str]
    ) -> tuple[list[Float["n m"]], list[Float["n #obj"]]]:
        """Load training data from CSV files and return tensors."""
        dfs = [pd.read_csv(f) for f in files]

        drop_cols = (
            ["rxn_id"] + self.config.objective_columns + self.config.exclude_columns
        )
        Xs_train = [
            Tensor(df.drop(columns=drop_cols).values).to(**tkwargs) for df in dfs
        ]
        ys_train = [
            Tensor(df[self.config.objective_columns].values).to(**tkwargs) for df in dfs
        ]

        for i, X in enumerate(Xs_train):
            if X.shape[0] != self.config.n_particles:
                self.log.critical(
                    f"Iteration {i+1}: expected {self.config.n_particles} "
                    f"rows, got {X.shape[0]}"
                )
                raise ValueError("Mismatch between swarm size and training rows.")
        return Xs_train, ys_train

    def _bootstrap_swarm_state(self, X0: Float["n m"], y0: Float["n #obj"]) -> None:
        """Bootstrap the swarm state with initial positions and scores."""
        scores = y0.sum(dim=1)  # TODO: adapt it to any scalarisation
        self.swarm.scores.append(scores)
        self.swarm.raw_scores.append(y0)

        best_idx = scores.argmax()
        self.swarm.global_best = scores[best_idx].item()
        self.swarm.global_best_position = X0[best_idx].to(**tkwargs)

        for idx, p in enumerate(self.swarm.particles):
            p.pbest_score = scores[idx].item()

        self.log.info(
            f"Step 0 (initialisation) - best score {self.swarm.global_best:.2f}"
        )

    def _train_initial_gps(self):
        """Train GPs with initial positions and scores."""
        X_train = self.swarm.positions.detach().clone().to(**tkwargs)
        y_train = self.swarm.raw_scores[0].detach().clone().to(**tkwargs)
        gps, _ = gp.get_trained_gp(X_train, y_train, model_config=self.gp_config)
        return gps, X_train, y_train

    def _iterative_optimisation(
        self,
        gps,
        X_train: Float["n m"],
        y_train: Float["n #obj"],
        Xs_train: list[Float["n m"]],
        ys_train: list[Float["n #obj"]],
    ) -> None:
        """Iteratively optimise the swarm based on experimental results."""
        for k in range(1, self.config.iteration_number + 1):
            self.swarm = acqf_manager(
                self.algo,
                swarm=self.swarm,
                pso_params=namedtuple("PSOParams", self.pso_params.keys())(
                    **self.pso_params
                ),
                gps=gps,
                X_train=X_train,
                y_train=y_train,
                seed=self.config.seed,
                iteration=k,
            )
            self.swarm.update_idx_explored()

            # break when we have generated the last plate’s suggestions
            if k + 1 == self.config.iteration_number:
                break

            # update with experimental results from plate k
            self._update_from_experiment(k, Xs_train[k], ys_train[k])
            X_train = torch.cat((X_train, Xs_train[k]))
            y_train = torch.cat(
                (y_train, self.swarm.raw_scores[k].detach().clone().to(**tkwargs))
            )
            gps, _ = gp.get_trained_gp(X_train, y_train, model_config=self.gp_config)

    def _update_from_experiment(self, k: int, X_i: Float["n m"], y_i: Float["n #obj"]) -> None:
        """Update the swarm state with experimental results from plate k."""
        scores = y_i.sum(dim=1)
        self.swarm.scores.append(scores)
        self.swarm.raw_scores.append(y_i)

        for idx, particle in enumerate(self.swarm.particles):
            score = scores[idx].item()
            if score >= particle.pbest_score:
                particle.pbest = particle.position
                particle.pbest_score = score
            if score >= self.swarm.global_best:
                self.swarm.global_best_position = particle.position
                self.swarm.global_best = score
        self.log.info(f"Step {k} - best score {self.swarm.global_best:.2f}")

    def _write_final_suggestions(self) -> None:
        """Write the final suggestions to a CSV file."""
        suggestions = self.swarm.idx_explored[-1]
        self.log.info(
            f"Iteration {self.config.iteration_number} - {len(suggestions)} suggestions"
        )
        self.log.info(f"Saving suggestions to {self.config.pso_suggestions_path}")

        df = self.config.raw_df.iloc[suggestions]
        file_path = os.path.join(
            self.config.pso_suggestions_path,
            self.config.pso_suggestions_format.format(self.config.iteration_number),
        )
        df.to_csv(file_path, index=False)
        self.log.info(
            f"Iteration {self.config.iteration_number}: {len(df)} suggestions "
            f"-> {file_path}"
        )

    def optimise(self) -> None:
        """Optimise the objective function."""
        # Compute scores of the initial positions
        scores, raw_scores = self.objective_function(
            self.swarm.positions.detach().cpu(), **self.obj_func_params
        )
        self.swarm.scores.append(scores)  # log the scores
        self.swarm.raw_scores.append(raw_scores)  # log the raw_scores

        global_best_idx = scores.argmax()
        self.swarm.global_best = scores[global_best_idx].item()
        self.swarm.global_best_position = self.swarm.positions[global_best_idx]
        self.log.info(
            f"Step 0 (initialisation) - Swarm best score: {self.swarm.global_best:.2f}"
        )

        # Train GPs models with initial positions and raw scores
        X_train = self.swarm.positions.detach().clone().to(**tkwargs)
        y_train = self.swarm.raw_scores[0].detach().clone().to(**tkwargs)
        gps, _ = gp.get_trained_gp(X_train, y_train, model_config=self.gp_config)

        # Main loop
        for k in range(1, self.n_iter):
            # Move the swarm
            self.swarm = acqf_manager(
                self.algo,
                swarm=self.swarm,
                pso_params=namedtuple("PSOParams", self.pso_params.keys())(
                    **self.pso_params
                ),
                gps=gps,
                X_train=X_train,
                y_train=y_train,
                seed=self.config.seed,
                iteration=k,
            )
            self.swarm.update_idx_explored()
            # Update the scores of the swarm
            self.update_swarm_scores()

            self.log.info(f"Step {k} - Swarm best score: {self.swarm.global_best:.2f}")

            # Update training data for gp and retrain the model
            X_train = torch.cat((X_train, self.swarm.positions))
            y_train = torch.cat(
                (y_train, self.swarm.raw_scores[k].detach().clone().to(**tkwargs))
            )

            gps, _ = gp.get_trained_gp(X_train, y_train, model_config=self.gp_config)

        self.log.info("Optimisation done!")

    def update_swarm_scores(self) -> None:
        """Update the scores of the swarm particles."""
        scores = []
        raw_scores = []

        for particle in self.swarm.particles:
            # Evaluate the objective function for the current particle position
            particle_score, particle_raw_score = self.objective_function(
                particle.position.unsqueeze(0).detach().cpu(), **self.obj_func_params
            )
            particle_score = particle_score.item()
            particle_raw_score = particle_raw_score.squeeze().tolist()

            # Evaluate the objective function for the personal best position
            pbest_score, _ = self.objective_function(
                particle.pbest.unsqueeze(0).detach().cpu(), **self.obj_func_params
            )

            # Update scores
            scores.append(particle_score)
            raw_scores.append(particle_raw_score)

            # Update personal best if the current score is better
            if particle_score >= pbest_score:
                particle.pbest = particle.position

            # Update global best if the current score is better
            if particle_score >= self.swarm.global_best:
                self.swarm.global_best_position = particle.position
                self.swarm.global_best = particle_score

        self.swarm.scores.append(Tensor(scores))
        self.swarm.raw_scores.append(Tensor(raw_scores))
