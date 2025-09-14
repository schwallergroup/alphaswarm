"""
Test cases for the objective functions
"""

import unittest

import torch
from alphaswarm.objective_functions import MultiObjective, SingleObjective
from torch import Tensor


class TestSingleObjective(unittest.TestCase):
    def setUp(self):
        self.X_norm = Tensor([[0.1, 0.1], [0.1, 0.3], [0.5, 0.6], [0.7, 0.8]])
        # Single objective
        self.y_target = Tensor([0.3, 0.2, 0.1, 0.0])
        self.single_obj = SingleObjective(self.X_norm, self.y_target)

    def test_get_indices(self):
        x = Tensor([[0.1, 0.3], [0.5, 0.6]])
        indices = self.single_obj.get_indices(x)
        self.assertTrue(torch.equal(indices, Tensor([1, 2])))

    def test_target(self):
        x = Tensor([[0.1, 0.3], [0.5, 0.6]])
        target = self.single_obj.target(x)
        self.assertTrue(torch.equal(target, Tensor([0.2, 0.1])))

    def test_single_objective(self):
        X_obs = Tensor([[0.1, 0.1], [0.1, 0.3], [0.7, 0.8]])
        single_obj, _ = self.single_obj.single_objective(X_obs)
        self.assertTrue(
            torch.allclose(single_obj, Tensor([0.3, 0.2, 0.0]), atol=1e-4, rtol=1e-4)
        )


class TestMultiObjective(unittest.TestCase):
    def setUp(self):
        self.X_norm = Tensor([[0.1, 0.1], [0.1, 0.3], [0.5, 0.6], [0.7, 0.8]])
        self.y_target = Tensor([[0.1, 0.0], [0.3, 0.4], [0.1, 0.2], [1.0, 0.9]])
        self.multiobj = MultiObjective(self.X_norm, self.y_target)
        self.X_obs = Tensor([[0.1, 0.3], [0.5, 0.6]])
        self.utopian_pt = Tensor([2.0, 2.0])

    def test_get_indices(self):
        indices = self.multiobj.get_indices(self.X_obs)
        self.assertTrue(torch.equal(indices, Tensor([1, 2])))

    def test_target(self):
        target = self.multiobj.target(self.X_obs)
        self.assertTrue(torch.allclose(target, Tensor([[0.3, 0.4], [0.1, 0.2]])))

    def test_weighted_sum(self):
        weights = [0.3, 0.7]
        results, _ = self.multiobj.weighted_sum(self.X_obs, weights=weights)
        self.assertTrue(torch.allclose(results, Tensor([0.3700, 0.1700])))

    def test_weighted_power(self):
        weights = [0.3, 0.7]
        p = 2
        results, _ = self.multiobj.weighted_power(self.X_obs, weights=weights, p=p)
        self.assertTrue(torch.allclose(results, Tensor([0.1390, 0.0310])))

    def test_weighted_norm(self):
        weights = [0.2, 0.8]
        p = 2
        results, _ = self.multiobj.weighted_norm(self.X_obs, weights=weights, p=p)
        self.assertTrue(
            torch.allclose(results, Tensor([0.3821, 0.1844]), atol=1e-4, rtol=1e-4)
        )

    def test_weighted_product(self):
        weights = [0.2, 0.8]
        results, _ = self.multiobj.weighted_product(self.X_obs, weights=weights)
        self.assertTrue(
            torch.allclose(results, Tensor([0.3776, 0.1741]), atol=1e-4, rtol=1e-4)
        )

    def test_chebeyshev_function(self):
        weights = [0.2, 0.8]
        results, _ = self.multiobj.chebyshev_function(
            self.X_obs, weights=weights, utopian_pt=self.utopian_pt
        )
        self.assertTrue(
            torch.allclose(results, Tensor([0.3400, 0.3800]), atol=1e-4, rtol=1e-4)
        )

    def test_augmented_chebeyshev(self):
        weights = [0.2, 0.8]
        alpha = 0.0001
        results, _ = self.multiobj.augmented_chebyshev(
            self.X_obs, weights=weights, utopian_pt=self.utopian_pt, alpha=alpha
        )
        self.assertTrue(
            torch.allclose(results, Tensor([0.3403, 0.3804]), atol=1e-4, rtol=1e-4)
        )

    def test_modified_chebeyshev(self):
        weights = [0.2, 0.8]
        alpha = 0.01
        results, _ = self.multiobj.modified_chebyshev(
            self.X_obs, weights=weights, utopian_pt=self.utopian_pt, alpha=alpha
        )
        self.assertTrue(
            torch.allclose(results, Tensor([0.3730, 0.4130]), atol=1e-4, rtol=1e-4)
        )


if __name__ == "__main__":
    unittest.main()
