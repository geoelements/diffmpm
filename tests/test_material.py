import jax.numpy as jnp
import pytest
from diffmpm.materials import LinearElastic, SimpleMaterial

material_dstrain_stress_targets = [
    (
        SimpleMaterial({"E": 10, "density": 1}),
        jnp.ones((1, 6, 1)),
        jnp.ones((1, 6, 1)) * 10,
    ),
    (
        LinearElastic({"density": 1, "youngs_modulus": 10, "poisson_ratio": 1}),
        jnp.ones((1, 6, 1)),
        jnp.array([-10, -10, -10, 2.5, 2.5, 2.5]).reshape(1, 6, 1),
    ),
    (
        LinearElastic({"density": 1000, "youngs_modulus": 1e7, "poisson_ratio": 0.3}),
        jnp.array([0.001, 0.0005, 0, 0, 0, 0]).reshape(1, 6, 1),
        jnp.array([1.63461538461538e4, 12500, 0.86538461538462e4, 0, 0, 0]).reshape(
            1, 6, 1
        ),
    ),
    (
        LinearElastic({"density": 1000, "youngs_modulus": 1e7, "poisson_ratio": 0.3}),
        jnp.array([0.001, 0.0005, 0, 0.00001, 0, 0]).reshape(1, 6, 1),
        jnp.array(
            [1.63461538461538e4, 12500, 0.86538461538462e4, 3.84615384615385e01, 0, 0]
        ).reshape(1, 6, 1),
    ),
]


@pytest.mark.parametrize("material, dstrain, target", material_dstrain_stress_targets)
def test_compute_stress(material, dstrain, target):
    stress = material.compute_stress(dstrain)
    assert jnp.allclose(stress, target)
