import unittest

from alphaswarm.utils.moo_utils import (
    get_pareto_points,
    hypervolume,
    hypervolume_improvement,
    hypervolume_regret,
    igd,
    igd_plus,
    igd_plus_distance_maximisation,
    log_hypervolume_regret,
)
from torch import Tensor


class TestMooUtils(unittest.TestCase):
    def setUp(self):
        self.points = Tensor([[0.2, 0.3], [2.1, 4.2], [5.3, 1.3]])
        self.ref_point = Tensor([0, 0])

    def test_hypervolume(self):
        hv = hypervolume(self.points, ref_point=self.ref_point)
        self.assertAlmostEqual(hv, 12.98, places=2)

    def test_hypervolume_improvement(self):
        train_y = Tensor([[1, 0], [0.5, 0.5], [0, 1], [1.5, 0.75]])
        ref_point = Tensor([0, 0])
        new_points = Tensor([[2, 2], [2, 3]])
        delta = hypervolume_improvement(
            ref_point=ref_point, Y=train_y, Y_new=new_points
        )
        self.assertAlmostEqual(delta, 4.875, places=3)

    def test_get_pareto_front(self):
        pareto_points = get_pareto_points(self.points)
        self.assertTrue(pareto_points.shape[0] == 2)
        self.assertTrue((pareto_points == self.points[[1, 2]]).all())

    def test_hypervolume_regret(self):
        hv_optimal = 30.0
        hv_achieved = 20.0
        hv_regret = hypervolume_regret(hv_achieved, hv_optimal)
        self.assertTrue(hv_regret == 10.0)

    def test_log_hypervolume_regret(self):
        hv_optimal = 30.0
        hv_achieved = 20.0
        hv_regret = log_hypervolume_regret(hv_achieved, hv_optimal)
        self.assertTrue(round(hv_regret, 2) == 2.30)

    def test_igd(self):
        pf = Tensor([[2.1, 4.2], [5.3, 1.3]])
        points = Tensor([[0.2, 0.3], [1.1, 2.2], [2.3, 0.3]])
        igd_score = igd(pf_true=pf, pf_approx=points)
        self.assertAlmostEqual(igd_score, 5.545, places=2)

    def test_igd_plus_distance_maximisation(self):
        pf = Tensor([[2.1, 4.2], [5.3, 1.3]])
        points = Tensor([[0.2, 0.3], [1.1, 2.2], [2.3, 0.3]])

        distance = igd_plus_distance_maximisation(
            true_point=pf[0], approx_point=points[0]
        )
        self.assertAlmostEqual(distance, 4.338, places=3)

    def test_igd_plus(self):
        pf = Tensor([[2.1, 4.2], [5.3, 1.3]])
        points = Tensor([[0.2, 0.3], [1.1, 2.2], [2.3, 0.3]])
        igd_score = igd_plus(pf_true=pf, pf_approx=points)
        self.assertAlmostEqual(igd_score, 2.699, places=3)


if __name__ == "__main__":
    unittest.main()
