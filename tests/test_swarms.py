import unittest

from alphaswarm.swarms import Particle, Swarm
from torch import Tensor


class TestParticle(unittest.TestCase):
    def test_particle(self):
        dim = 2
        position = Tensor([0.1, 0.2])
        velocity = Tensor([0.01, 0.02])
        pbest = Tensor([0.3, 0.1])
        particle = Particle(dim, position, velocity, pbest)
        self.assertEqual(particle.dim, dim)
        self.assertTrue((particle.position == position).all())
        self.assertTrue((particle.velocity == velocity).all())
        self.assertTrue((particle.pbest == pbest).all())


class TestSwarm(unittest.TestCase):
    def test_swarm(self):
        dim = 2
        n_particles = 2
        init_method = "sobol"
        X_norm = Tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        swarm = Swarm(dim, n_particles, init_method, X_norm)
        swarm.init_particles()
        self.assertEqual(swarm.dim, dim)
        self.assertEqual(len(swarm), n_particles)
        self.assertEqual(swarm.init_method, init_method)
        self.assertTrue(set(swarm.idx_explored[0]).issubset(set([0, 1, 2])))


if __name__ == "__main__":
    unittest.main()
