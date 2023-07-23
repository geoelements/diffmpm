import jax.numpy as jnp
import pytest
from diffmpm.materials import init_linear_elastic, init_simple
from diffmpm.particle import init_particle_state

particles_dstrain_stress_targets = [
    (
        init_particle_state(
            jnp.array([[0.5, 0.5]]).reshape(1, 1, 2),
            init_simple({"id": 0, "E": 10, "density": 1}),
            jnp.array([0]),
        ),
        jnp.ones((1, 6, 1)),
        jnp.ones((1, 6, 1)) * 10,
    ),
    (
        init_particle_state(
            jnp.array([[0.5, 0.5]]).reshape(1, 1, 2),
            init_linear_elastic(
                {"id": 0, "density": 1, "youngs_modulus": 10, "poisson_ratio": 1}
            ),
            jnp.array([0]),
        ),
        jnp.ones((1, 6, 1)),
        jnp.array([-10, -10, -10, 2.5, 2.5, 2.5]).reshape(1, 6, 1),
    ),
    (
        init_particle_state(
            jnp.array([[0.5, 0.5]]).reshape(1, 1, 2),
            init_linear_elastic(
                {"id": 0, "density": 1000, "youngs_modulus": 1e7, "poisson_ratio": 0.3}
            ),
            jnp.array([0]),
        ),
        jnp.array([0.001, 0.0005, 0, 0, 0, 0]).reshape(1, 6, 1),
        jnp.array([1.63461538461538e4, 12500, 0.86538461538462e4, 0, 0, 0]).reshape(
            1, 6, 1
        ),
    ),
    (
        init_particle_state(
            jnp.array([[0.5, 0.5]]).reshape(1, 1, 2),
            init_linear_elastic(
                {"id": 0, "density": 1000, "youngs_modulus": 1e7, "poisson_ratio": 0.3}
            ),
            jnp.array([0]),
        ),
        jnp.array([0.001, 0.0005, 0, 0.00001, 0, 0]).reshape(1, 6, 1),
        jnp.array(
            [1.63461538461538e4, 12500, 0.86538461538462e4, 3.84615384615385e01, 0, 0]
        ).reshape(1, 6, 1),
    ),
]


@pytest.mark.parametrize("particles, dstrain, target", particles_dstrain_stress_targets)
def test_compute_stress(particles, dstrain, target):
    particles = particles.replace(dstrain=dstrain)
    stress = particles.material.compute_stress(particles)
    assert jnp.allclose(stress, target)
