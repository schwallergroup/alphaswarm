"""Command line interface for running the PSO algorithm."""

import sys

import typer

from alphaswarm.configs import PSOBenchmarkConfig, PSOExperimentConfig
from alphaswarm.pso import PSO
from alphaswarm.utils.logger import Logger

log = Logger().log
app = typer.Typer()


@app.command()
def benchmark(config: str):
    """Load a `.toml`configuration file and run the benchmark."""
    try:
        benchmark_config = PSOBenchmarkConfig.from_toml(config)
        pso = PSO(benchmark_config)
        pso.optimise()
    except FileNotFoundError:
        log.error(f":open_file_folder: Benchmark config file `{config}` not found.")
        sys.exit(1)


@app.command()
def experimental(config: str):
    """Load a `.toml`configuration file and run the experimental suggestions."""
    try:
        exp_config = PSOExperimentConfig.from_toml(config)
        pso = PSO(exp_config)
        pso.suggest()
    except FileNotFoundError:
        log.error(f":open_file_folder: Experimental config file `{config}` not found.")
        sys.exit(1)


def main():
    """Main entry point for the CLI."""
    app()
