import unittest

import torch
from alphaswarm.utils.utils import (
    check_tensor,
    get_closest_features,
    get_closest_unknown_features,
    nadir_point,
    normalise_features,
    random_initialisation,
    sobol_sequence,
)
from torch import Tensor


class TestUtilsFunctions(unittest.TestCase):
    def test_normalise_features(self):
        X_features = Tensor([[1.0, 2.0], [3.0, 4.0]])
        normalised = normalise_features(X_features)
        expected = Tensor([[0.0, 0.0], [1.0, 1.0]])
        self.assertTrue(
            torch.allclose(normalised, expected),
            "The normalisation did not produce the expected result.",
        )
        self.assertEqual(
            normalised.shape,
            X_features.shape,
            "The shape of the normalised tensor is incorrect.",
        )

    def test_get_closest_features(self):
        x = Tensor([[0.4, 0.6]])
        feature_space = Tensor([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        closest = get_closest_features(x, feature_space)
        expected = Tensor([[0.5, 0.5]])
        self.assertTrue(
            torch.allclose(closest, expected),
            "The closest feature was not the expected one.",
        )
        self.assertEqual(
            closest.shape, x.shape, "The shape of the closest Tensor is incorrect."
        )

    def test_get_closest_unknown_features(self):
        x = Tensor([[0.4, 0.7]])
        feature_space = Tensor([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        known_indices = set([1])
        closest = get_closest_unknown_features(x, feature_space, known_indices)
        expected = Tensor([[1.0, 1.0]])
        self.assertTrue(
            torch.allclose(closest, expected),
            "The closest unknown feature was not the expected one.",
        )
        self.assertEqual(
            closest.shape, x.shape, "The shape of the closest Tensor is incorrect."
        )

    def test_check_tensor_valid(self):
        tensor = Tensor([1.0, 2.0, 3.0])
        try:
            check_tensor(tensor, (3, 3), 1, "test_tensor")
        except ValueError:
            self.fail("check_tensor raised ValueError unexpectedly!")

    def test_check_tensor_invalid_shape(self):
        tensor = Tensor([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            check_tensor(tensor, (3, 4), 1, "test_tensor")

    def test_check_tensor_invalid_dim(self):
        tensor = Tensor([[1.0, 2.0, 3.0]])
        with self.assertRaises(ValueError):
            check_tensor(tensor, (3, 3), 1, "test_tensor")

    def test_random_initialisation(self):
        dim = 3
        n_particles = 5
        result = random_initialisation(dim, n_particles)
        self.assertEqual(
            result.shape,
            (n_particles, dim),
            "random_initialisation did not produce the expected shape.",
        )

    def test_sobol_sequence(self):
        dim = 3
        n_particles = 5
        result = sobol_sequence(dim, n_particles)
        self.assertEqual(
            result.shape,
            (n_particles, dim),
            "sobol_sequence did not produce the expected shape.",
        )

    def test_nadir_point(self):
        Y = Tensor([[1.0, 2.0], [3.0, 4.0]])
        result = nadir_point(Y)
        expected = Tensor([1.0, 2.0])
        self.assertTrue(
            torch.allclose(result, expected),
            "nadir_point did not produce the expected result.",
        )
        self.assertEqual(
            result.shape, (2,), "nadir_point did not produce the expected shape."
        )


if __name__ == "__main__":
    unittest.main()
