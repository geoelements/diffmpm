import jax.numpy as jnp
import pytest

from diffmpm.element import Quadrilateral4Node
from diffmpm.materials import init_simple
from diffmpm.particle import init_particle_state
import diffmpm.particle as dpar


class TestParticles:
    @pytest.fixture
    def elements(self):
        return Quadrilateral4Node((1, 1), 1, (1.0, 1.0), [])

    @pytest.fixture
    def particles(self):
        loc = jnp.array([[0.5, 0.5], [0.5, 0.5]]).reshape(2, 1, 2)
        material = init_simple({"id": 0, "E": 1, "density": 1})
        return init_particle_state(loc, material, jnp.array([0, 0]))

    @pytest.mark.parametrize(
        "velocity_update, expected",
        [
            (False, jnp.array([[0.1, 0.1], [0.1, 0.1]]).reshape(2, 1, 2)),
            (True, jnp.array([[1, 1], [1, 1]]).reshape(2, 1, 2)),
        ],
    )
    def test_update_velocity(self, elements, particles, velocity_update, expected):
        dpar.update_natural_coords(particles, elements)
        elements.nodes = elements.nodes.replace(
            acceleration=elements.nodes.acceleration + 1
        )
        elements.nodes = elements.nodes.replace(velocity=elements.nodes.velocity + 1)
        particles = dpar.update_position_velocity(
            particles, elements, 0.1, velocity_update
        )
        assert jnp.allclose(particles.velocity, expected)

    def test_compute_strain(self, elements, particles):
        elements.nodes = elements.nodes.replace(
            velocity=jnp.array([[0, 1], [0, 2], [0, 3], [0, 4]]).reshape(4, 1, 2)
        )
        particles = dpar.update_natural_coords(particles, elements)
        particles = dpar.compute_strain(particles, elements, 0.1)
        assert jnp.allclose(
            particles.strain,
            jnp.array([[0, 0.2, 0, 0.1, 0, 0], [0, 0.2, 0, 0.1, 0, 0]]).reshape(
                2, 6, 1
            ),
        )
        assert jnp.allclose(particles.volumetric_strain_centroid, jnp.array([0.2]))

    def test_compute_volume(self, elements, particles):
        particles = dpar.compute_volume(particles, elements, elements.total_elements)
        assert jnp.allclose(particles.volume, jnp.array([0.5, 0.5]).reshape(2, 1, 1))

    def test_assign_traction(self, elements, particles):
        particles = dpar.compute_volume(particles, elements, elements.total_elements)
        traction = dpar.assign_traction(particles, jnp.array([0]), 1, 10)
        assert jnp.allclose(
            traction, jnp.array([[0, 7.071068], [0, 0]]).reshape(2, 1, 2)
        )

    def test_zero_traction(self, particles):
        particles = particles.replace(traction=particles.traction + 1)
        particles = dpar.zero_traction(particles)
        assert jnp.all(particles.traction == 0)
