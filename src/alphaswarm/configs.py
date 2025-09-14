"""Configuration classes for PSO and BO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import toml

from alphaswarm.utils.logger import Logger

log = Logger().log


@dataclass
class PSOBenchmarkConfig:
    """Configuration for the Particle Swarm Optimisation algorithm for benchmark."""

    file_path: str  # path to the dataset
    y_columns: list[str]  # column names of the target variables

    seed: int  # random seed
    n_iter: int  # max number of iterations including the initialisation

    n_particles: int  # number of particles
    init_method: str  # position initialisation method
    algo: str  # algorithm to update the swarm
    pso_params: dict[str, Any]  # parameters of the PSO algorithm

    objective_function: str  # type of objective function
    obj_func_params: dict[str, Any]  # parameters of the objective function

    model_config: dict[str, Any]  # configuration for the surrogate model

    exclude_columns: list[str] | None = (
        None  # columns to exclude from the dataset (optional)
    )

    def __post_init__(self):
        """Post-initialisation checks and data preparation."""
        self._validate_parameters()
        self._load_and_prepare_data()

    def _validate_parameters(self):
        """Validate the parameters of the configuration."""
        if self.n_iter <= 0:
            raise ValueError("n_iter must be greater than 0")

        if self.algo in ["alpha-pso", "pso"]:
            # Check PSO-specific parameters
            required_pso_keys = {"c_1", "c_2", "c_a", "w"}
            if not required_pso_keys.issubset(self.pso_params.keys()):
                raise ValueError(
                    f"pso_params must contain the following keys: {required_pso_keys}"
                )

        if "noise" not in self.obj_func_params:
            raise ValueError("obj_func_params must contain the key 'noise'")

        self.model_config = ModelConfig(**self.model_config)

    def _load_and_prepare_data(self):
        """Load the dataset and prepare features and target variables."""
        df = pd.read_csv(self.file_path)

        if not set(self.y_columns).issubset(df.columns):
            raise ValueError("y_columns must be in the columns of the dataset")

        if "Unnamed: 0" in df.columns:
            df.drop(columns="Unnamed: 0", axis=1, inplace=True)

        if self.exclude_columns:
            df.drop(columns=self.exclude_columns, inplace=True)

        df.drop(columns="rxn_id", inplace=True)
        self.X_features = df.drop(columns=self.y_columns).values
        self.y_target = df[self.y_columns].values

    @classmethod
    def from_toml(cls, path: str) -> PSOBenchmarkConfig:
        """Create a PSOBenchmarkConfig instance from a toml file."""
        with open(path) as f:
            config = toml.load(f)
            log.info(
                f":white_check_mark: Configuration for benchmarking loaded from {path}:"
            )
        return cls(**config)


@dataclass
class ModelConfig:
    """Configuration for the surrogate model."""

    kernel: str  # name of the kernel [MaternKernel, KMaternKernel]
    kernel_params: dict[str, Any] | str  # kernel hyperparameters
    training_iter: int = 1000  # number of training iterations

    def __post_init__(self):
        """Check parameter values."""
        if self.training_iter <= 0:
            raise ValueError("training_iter must be greater than 0")

        if self.kernel not in {"MaternKernel", "KMaternKernel"}:
            raise ValueError("kernel must be either 'MaternKernel' or 'KMaternKernel'")

        if self.kernel_params == "default":
            self.kernel_params = {
                "ls_prior1": 2.0,
                "ls_prior2": 0.2,
                "ls_prior3": 5.0,
                "out_prior1": 5.0,
                "out_prior2": 0.5,
                "out_prior3": 8.0,
                "noise_prior1": 1.5,
                "noise_prior2": 0.1,
                "noise_prior3": 5.0,
                "noise_constraint": 1e-5,
            }
        required_keys = {
            "noise_prior1",
            "noise_prior2",
            "noise_prior3",
            "noise_constraint",
        }
        if not required_keys.issubset(self.kernel_params.keys()):
            raise ValueError(
                f"kernel_params must contain the following keys: {required_keys}"
            )


@dataclass
class PSOExperimentConfig:
    """Config for a lab experiment."""

    chemical_space_file: str  # path to the chemical space

    seed: int  # random seed
    n_particles: int  # number of particles
    init_method: str  # position initialisation method
    algo: str  # algorithm to update the swarm
    pso_params: dict[str, Any]  # parameters of the PSO algorithm

    iteration_number: int  # current iteration number (starting from 1)

    model_config: dict[str, Any]  # configuration for the surrogate model

    objective_columns: list[str]  # column names of the target variables

    # ouput path for the PSO suggestions
    pso_suggestions_path: str = "data/experimental_campaigns/pso_plate_suggestions"
    pso_suggestions_format: str = (
        "PSO_plate_{}_suggestions.csv"  # format of the PSO suggestions
    )

    experimental_data_format: str = (
        "PSO_plate_{}_train.csv"  # format of the training data
    )
    experimental_data_path: str = (
        "data/experimental_campaigns/pso_training_data"  # path to the experimental data
    )

    exclude_columns: list[str] | None = (
        None  # columns to exclude from the chemical space (optional)
    )

    def __post_init__(self):
        """Post-initialisation checks and data preparation."""
        self._validate_parameters()
        self._load_and_prepare_data()

    def _validate_parameters(self):
        """Validate the parameters of the configuration."""
        if self.iteration_number <= 0:
            raise ValueError("iteration_number must be greater than 0")

        if self.algo in ["alpha-pso", "pso"]:
            # Check PSO-specific parameters
            required_pso_keys = {"c_1", "c_2", "c_a", "w"}
            if not required_pso_keys.issubset(self.pso_params.keys()):
                raise ValueError(
                    f"pso_params must contain the following keys: {required_pso_keys}"
                )

        self.model_config = ModelConfig(**self.model_config)

    def _load_and_prepare_data(self):
        """Load the chemical space."""
        df = pd.read_csv(self.chemical_space_file)

        if "rxn_id" not in df.columns:
            raise ValueError("rxn_id must be in the columns of the chemical space")

        if "Unnamed: 0" in df.columns:
            df.drop(columns="Unnamed: 0", axis=1, inplace=True)

        self.raw_df = df.copy()

        if self.exclude_columns:
            df.drop(columns=self.exclude_columns, inplace=True)

        df.drop(columns="rxn_id", inplace=True)
        self.X_features = df.values
        self.features_names = df.columns.values

    @classmethod
    def from_toml(cls, path: str) -> PSOExperimentConfig:
        """Create a PSOExperimentConfig instance from a toml file."""
        with open(path) as f:
            config = toml.load(f)
            log.info(
                f":white_check_mark: Configuration for experimental loaded from {path}:"
            )
        return cls(**config)
