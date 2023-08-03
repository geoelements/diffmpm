import jax.numpy as jnp
from functools import partial
import pytest
from jax import vmap

from diffmpm.element import Quad4N
from diffmpm.materials import init_simple
from diffmpm.particle import init_particle_state
import diffmpm.particle as dpar


class TestParticles:
    elementor = Quad4N(total_elements=1)

    @pytest.fixture
    def elements(self):
        return self.elementor.init_state((1, 1), 1, (1.0, 1.0), [])

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
        dpar.update_natural_coords(particles, elements, self.elementor)
        elements.nodes = elements.nodes.replace(
            acceleration=elements.nodes.acceleration + 1
        )
        elements.nodes = elements.nodes.replace(velocity=elements.nodes.velocity + 1)
        updated = dpar._update_particle_position_velocity(
            Quad4N,
            particles.loc,
            particles.velocity,
            particles.momentum,
            particles.mass,
            particles.reference_loc,
            vmap(partial(self.elementor.id_to_node_ids, 1))(particles.element_ids),
            elements.nodes.velocity,
            elements.nodes.acceleration,
            velocity_update,
            0.1,
        )
        assert jnp.allclose(updated["velocity"], expected)

    def test_compute_strain(self, elements, particles):
        elements.nodes = elements.nodes.replace(
            velocity=jnp.array([[0, 1], [0, 2], [0, 3], [0, 4]]).reshape(4, 1, 2)
        )
        particles = dpar.update_natural_coords(particles, elements, self.elementor)
        updated = dpar._compute_strain(
            particles.strain,
            particles.reference_loc,
            particles.loc,
            particles.volumetric_strain_centroid,
            particles.nparticles,
            vmap(partial(self.elementor.id_to_node_ids, 1))(particles.element_ids),
            elements.nodes.loc,
            elements.nodes.velocity,
            Quad4N,
            0.1,
        )
        assert jnp.allclose(
            updated["strain"],
            jnp.array([[0, 0.2, 0, 0.1, 0, 0], [0, 0.2, 0, 0.1, 0, 0]]).reshape(
                2, 6, 1
            ),
        )
        assert jnp.allclose(updated["volumetric_strain_centroid"], jnp.array([0.2]))

    def test_compute_volume(self, elements, particles):
        props = dpar._compute_particle_volume(
            particles.element_ids,
            self.elementor.total_elements,
            elements.volume,
            particles.volume,
            particles.size,
            particles.mass,
            particles.density,
        )
        assert jnp.allclose(props["volume"], jnp.array([0.5, 0.5]).reshape(2, 1, 1))

    @pytest.mark.skip()
    def test_assign_traction(self, elements, particles):
        particles = dpar.compute_volume(particles, elements, elements.total_elements)
        traction = dpar.assign_traction(particles, jnp.array([0]), 1, 10)
        assert jnp.allclose(
            traction, jnp.array([[0, 7.071068], [0, 0]]).reshape(2, 1, 2)
        )

    def test_zero_traction(self, particles):
        particles = particles.replace(traction=particles.traction + 1)
        traction = dpar._zero_traction(particles.traction)
        assert jnp.all(traction == 0)
