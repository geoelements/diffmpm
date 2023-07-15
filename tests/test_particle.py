import jax.numpy as jnp
import pytest

from diffmpm.element import Quadrilateral4Node
from diffmpm.materials import SimpleMaterial
from diffmpm.particle import Particles


class TestParticles:
    @pytest.fixture
    def elements(self):
        return Quadrilateral4Node((1, 1), 1, (1.0, 1.0), [])

    @pytest.fixture
    def particles(self):
        loc = jnp.array([[0.5, 0.5], [0.5, 0.5]]).reshape(2, 1, 2)
        material = SimpleMaterial({"E": 1, "density": 1})
        return Particles(loc, material, jnp.array([0, 0]))

    @pytest.mark.parametrize(
        "velocity_update, expected",
        [
            (False, jnp.array([[0.1, 0.1], [0.1, 0.1]]).reshape(2, 1, 2)),
            (True, jnp.array([[1, 1], [1, 1]]).reshape(2, 1, 2)),
        ],
    )
    def test_update_velocity(self, elements, particles, velocity_update, expected):
        particles.update_natural_coords(elements)
        elements.nodes.acceleration += 1
        elements.nodes.velocity += 1
        particles.update_position_velocity(elements, 0.1, velocity_update)
        assert jnp.allclose(particles.velocity, expected)

    def test_compute_strain(self, elements, particles):
        elements.nodes.velocity = jnp.array([[0, 1], [0, 2], [0, 3], [0, 4]]).reshape(
            4, 1, 2
        )
        particles.update_natural_coords(elements)
        particles.compute_strain(elements, 0.1)
        assert jnp.allclose(
            particles.strain,
            jnp.array([[0, 0.2, 0, 0.1, 0, 0], [0, 0.2, 0, 0.1, 0, 0]]).reshape(
                2, 6, 1
            ),
        )
        assert jnp.allclose(particles.volumetric_strain_centroid, jnp.array([0.2]))

    def test_compute_volume(self, elements, particles):
        particles.compute_volume(elements, elements.total_elements)
        assert jnp.allclose(particles.volume, jnp.array([0.5, 0.5]).reshape(2, 1, 1))

    def test_assign_traction(self, elements, particles):
        particles.compute_volume(elements, elements.total_elements)
        particles.assign_traction(jnp.array([0]), 1, 10)
        assert jnp.allclose(
            particles.traction, jnp.array([[0, 7.071068], [0, 0]]).reshape(2, 1, 2)
        )

    def test_zero_traction(self, particles):
        particles.traction += 1
        particles.zero_traction()
        assert jnp.all(particles.traction == 0)
