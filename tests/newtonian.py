import jax.numpy as jnp
import pytest
from diffmpm.constraint import Constraint
from diffmpm.element import Quadrilateral4Node
from diffmpm.materials import Newtonian
from diffmpm.node import Nodes
from diffmpm.particle import Particles

particles_element_targets = [
    (
        Particles(
            jnp.array([[0.5, 0.5]]).reshape(1, 1, 2),
            Newtonian(
                {
                    "density": 1000,
                    "bulk_modulus": 8333333.333333333,
                    "dynamic_viscosity": 8.9e-4,
                }
            ),
            jnp.array([0]),
        ),
        Quadrilateral4Node(
            (1, 1),
            1,
            (4.0, 4.0),
            [(0, Constraint(0, 0.02)), (0, Constraint(1, 0.03))],
            Nodes(4, jnp.array([-2, -2, 2, -2, -2, 2, 2, 2]).reshape((4, 1, 2))),
        ),
        jnp.array(
            [
                -52083.3333338896,
                -52083.3333355583,
                -52083.3333305521,
                -0.0000041719,
                0,
                0,
            ]
        ).reshape(1, 6, 1),
    ),
    (
        Particles(
            jnp.array([[0.5, 0.5]]).reshape(1, 1, 2),
            Newtonian(
                {
                    "density": 1000,
                    "bulk_modulus": 8333333.333333333,
                    "dynamic_viscosity": 8.9e-4,
                    "incompressible": True,
                }
            ),
            jnp.array([0]),
        ),
        Quadrilateral4Node(
            (1, 1),
            1,
            (4.0, 4.0),
            [(0, Constraint(0, 0.02)), (0, Constraint(1, 0.03))],
            Nodes(4, jnp.array([-2, -2, 2, -2, -2, 2, 2, 2]).reshape((4, 1, 2))),
        ),
        jnp.array(
            [
                -0.0000033375,
                -0.00000500625,
                0,
                -0.0000041719,
                0,
                0,
            ]
        ).reshape(1, 6, 1),
    ),
]


@pytest.mark.parametrize(
    "particles, element, target",
    particles_element_targets,
)
def test_compute_stress(particles, element, target):
    dt = 1
    particles.update_natural_coords(element)
    if element.constraints:
        element.apply_boundary_constraints()
    particles.compute_strain(element, dt)
    stress = particles.material.compute_stress(particles)
    assert jnp.allclose(stress, target)


def test_init():
    with pytest.raises(KeyError):
        Newtonian({"dynamic_viscosity": 1, "density": 1})
