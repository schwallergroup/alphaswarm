"""Gaussian Process model implemented using GPyTorch and BoTorch."""

from collections import namedtuple

import gpytorch
import torch
from alphaswarm.configs import ModelConfig
from alphaswarm.utils.tensor_types import Float, typecheck
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.utils.transforms import standardize
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.kernels.keops import MaternKernel as KMaternKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior
from sklearn.preprocessing import StandardScaler
from torch import Tensor

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

kernels_map = {
    "MaternKernel": MaternKernel,
    "KMaternKernel": KMaternKernel,
}


class ExactGPModel(gpytorch.models.ExactGP):
    """GP model from GPyTorch."""

    @typecheck
    def __init__(
        self,
        train_x: Float["n m"],
        train_y: Float["n _"],
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        config: ModelConfig,
    ) -> None:
        """Initialise the GP model.

        Args:
            train_x: `n` training data with `m` features.
            train_y: `n` training targets with 1 objective.
            likelihood: Gaussian likelihood.
            config: Configuration for the surrogate model.
        """
        super().__init__(train_x, train_y, likelihood)
        self.kernel = kernels_map[config.kernel]
        self.kernel_params = config.kernel_params

        self.mean_module = gpytorch.means.ConstantMean()

        dim = train_x.shape[-1]

        kernels = self.kernel(
            ard_num_dims=dim,
            lengthscale_prior=GammaPrior(
                self.kernel_params["ls_prior1"], self.kernel_params["ls_prior2"]
            ),
        )
        self.covar_module = ScaleKernel(
            kernels,
            outputscale_prior=GammaPrior(
                self.kernel_params["out_prior1"], self.kernel_params["out_prior2"]
            ),
        )

        self.covar_module.base_kernel.lengthscale = self.kernel_params["ls_prior3"]

    def forward(
        self, x: Float["batch dim"]
    ) -> gpytorch.distributions.MultivariateNormal:
        """Forward pass.

        Args:
            x: `batch` data with `dim` features.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GP:
    """Gaussian Process model."""

    def __init__(
        self,
        train_X: Float["n m"],
        train_y: Float["n 1"],
        model_config: ModelConfig,
    ) -> None:
        """Initialise the GP model.

        Args:
            train_X: `n` training data with `m` features.
            train_y: `n` training targets with 1 objective.
            model_config: Configuration for the surrogate model.
        """
        self.train_X = train_X
        self.train_y = train_y
        self.model_config = model_config
        self.kernel = model_config.kernel
        self.kernel_params = model_config.kernel_params
        self.training_iter = model_config.training_iter

    def train(self) -> tuple[ExactGPModel, gpytorch.likelihoods.GaussianLikelihood]:
        """Train the GP model for a single objective."""
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            GammaPrior(
                self.kernel_params["noise_prior1"], self.kernel_params["noise_prior2"]
            )
        )

        likelihood.noise = self.kernel_params["noise_prior3"]
        model = ExactGPModel(
            train_x=self.train_X,
            train_y=self.train_y,
            likelihood=likelihood,
            config=self.model_config,
        ).to(**tkwargs)
        model.likelihood.noise_covar.register_constraint(
            "raw_noise", GreaterThan(self.kernel_params["noise_constraint"])
        )

        # Start training
        model.train()
        likelihood.train()

        optimiser = torch.optim.Adam([{"params": model.parameters()}], lr=0.1)
        mll = ExactMarginalLogLikelihood(likelihood, model)

        for _ in range(self.training_iter):
            optimiser.zero_grad()
            output = model(self.train_X)
            loss = -mll(output, self.train_y.squeeze(-1).to(**tkwargs))
            loss.backward()
            optimiser.step()

        model.eval()
        likelihood.eval()

        return model, likelihood


def get_trained_gp(
    train_X: Float["n m"], train_Y: Float["n n_obj"], model_config: ModelConfig
) -> tuple[ModelListGP, list]:
    """Get trained GP model over all objectives.

    Args:
        train_X: `n` training data with `m` features.
        train_Y: `n` unstandardised training targets with `n_obj` objectives.
        model_config: Configuration for the surrogate model.

    Returns:
        Trained GP model and likelihoods.
    """
    individual_models = []
    likelihoods = []
    train_Y = standardize(train_Y)

    for i in range(train_Y.shape[-1]):
        train_y_i = train_Y[..., i : i + 1]
        gp_model = GP(train_X, train_y_i, model_config=model_config)
        gp, likelihood = gp_model.train()

        model_i = SingleTaskGP(
            train_X=train_X,
            train_Y=train_y_i,
            covar_module=gp.covar_module,
            likelihood=likelihood,
        )

        model_i.eval()
        likelihood.eval()

        individual_models.append(model_i)
        likelihoods.append(likelihood)

    model_list = ModelListGP(*individual_models)  # for independent outputs

    return model_list, likelihoods


def get_models_pred(gps: ModelListGP, X_pred: Float["n m"], scaler: StandardScaler):
    """Get the predictions of the model.

    Args:
        gps: Trained GP model list.
        X_pred: `n` points of `m`-dimension to predict.
        scaler: StandardScaler object to inverse transform the predictions.

    Returns:
        Mean and variance of the predictions.
    """
    mean = Tensor(scaler.inverse_transform(gps.posterior(X_pred).mean.detach().cpu()))
    variance = Tensor(
        scaler.inverse_transform(gps.posterior(X_pred).variance.detach().cpu())
    )

    return namedtuple("Predictions", ["mean", "variance"])(mean, variance)
