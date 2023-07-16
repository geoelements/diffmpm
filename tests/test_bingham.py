import pytest
import jax.numpy as jnp
from diffmpm.material import Bingham
from diffmpm.particle import Particles
from diffmpm.element import Quadrilateral4Node
from diffmpm.constraint import Constraint
from diffmpm.node import Nodes

particles_element_targets = [
    (
        Particles(
            jnp.array([[0.5, 0.5]]).reshape(1, 1, 2),
            (
                Bingham(
                    {
                        "density": 1000,
                        "youngs_modulus": 1.0e7,
                        "poisson_ratio": 0.3,
                        "tau0": 771.8,
                        "mu": 0.0451,
                        "critical_shear_rate": 0.2,
                        "ndim": 2,
                    }
                )
            ),
            jnp.array([0]),
        ),
        Quadrilateral4Node(
            (1, 1),
            1,
            (4.0, 4.0),
            [],
            Nodes(4, jnp.array([-2, -2, 2, -2, -2, 2, 2, 2]).reshape((4, 1, 2))),
        ),
        jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape((6, 1)),
    ),
    (
        Particles(
            jnp.array([[0.5, 0.5]]).reshape(1, 1, 2),
            (
                Bingham(
                    {
                        "density": 1000,
                        "youngs_modulus": 1.0e7,
                        "poisson_ratio": 0.3,
                        "tau0": 771.8,
                        "mu": 0.0451,
                        "critical_shear_rate": 0.2,
                        "ndim": 2,
                    }
                )
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
        jnp.array([-52083.3333333333, -52083.3333333333, 0.0, 0.0, 0.0, 0.0]).reshape(
            (6, 1)
        ),
    ),
    (
        Particles(
            jnp.array([[0.5, 0.5]]).reshape(1, 1, 2),
            (
                Bingham(
                    {
                        "density": 1000,
                        "youngs_modulus": 1.0e7,
                        "poisson_ratio": 0.3,
                        "tau0": 200.0,
                        "mu": 200.0,
                        "critical_shear_rate": 0.2,
                        "ndim": 2,
                    }
                )
            ),
            jnp.array([0]),
        ),
        Quadrilateral4Node(
            (1, 1),
            1,
            (4.0, 4.0),
            [(0, Constraint(0, 2.0)), (0, Constraint(1, 3.0))],
            Nodes(4, jnp.array([-2, -2, 2, -2, -2, 2, 2, 2]).reshape((4, 1, 2))),
        ),
        jnp.array(
            [-5208520.35574006, -5208613.86694342, 0.0, -233.778008402801, 0.0, 0.0]
        ).reshape((6, 1)),
    ),
    (
        Particles(
            jnp.array([[0.5, 0.5]]).reshape(1, 1, 2),
            (
                Bingham(
                    {
                        "density": 1000,
                        "youngs_modulus": 1.0e7,
                        "poisson_ratio": 0.3,
                        "tau0": 200.0,
                        "mu": 200.0,
                        "critical_shear_rate": 0.2,
                        "ndim": 2,
                        "incompressible": True,
                    }
                )
            ),
            jnp.array([0]),
        ),
        Quadrilateral4Node(
            (1, 1),
            1,
            (4.0, 4.0),
            [(0, Constraint(0, 2.0)), (0, Constraint(1, 3.0))],
            Nodes(4, jnp.array([-2, -2, 2, -2, -2, 2, 2, 2]).reshape((4, 1, 2))),
        ),
        jnp.array(
            [-187.0224067222, -280.5336100834, 0.0, -233.778008402801, 0.0, 0.0]
        ).reshape((6, 1)),
    ),
]

@pytest.mark.parametrize(
    "particles, element, target",
    particles_element_targets,
)
def test_compute_stress(particles, element, target):
    particles.update_natural_coords(element)
    if element.constraints:
        element.apply_boundary_constraints()
    particles.compute_strain(element, 1.0)
    stress = particles.material.compute_stress(None, particles, {"pressure": jnp.zeros(1)})
    assert jnp.allclose(stress, target)


def test_key_not_present_in_material_properties():
    with pytest.raises(KeyError):
        material = Bingham(
            {
                "density": 1000,
                "youngs_modulus": 1.0e7,
                "poisson_ratio": 0.3,
                "tau0": 771.8,
                "critical_shear_rate": 0.2,
                "ndim": 2,
            }
        )


@pytest.mark.parametrize(
    "particles, element, target, state_vars",
    [
        (Particles(
            jnp.array([[0.5, 0.5, 0.5, 0.5]]).reshape(2, 1, 2),
            (
                Bingham(
                    {
                        "density": 1000,
                        "youngs_modulus": 1.0e7,
                        "poisson_ratio": 0.3,
                        "tau0": 200.0,
                        "mu": 200.0,
                        "critical_shear_rate": 0.2,
                        "ndim": 2,
                    }
                )
            ),
            jnp.array([0,0]),
        ),
        Quadrilateral4Node(
            (1, 1),
            1,
            (4.0, 4.0),
            [(0, Constraint(0, 2.0)), (0, Constraint(1, 3.0))],
            Nodes(4, jnp.array([-2, -2, 2, -2, -2, 2, 2, 2]).reshape((4, 1, 2))),
        ),
        jnp.array(
            [
                -5208520.35574006,
                -5208613.86694342,
                0.0,
                -233.778008402801,
                0.0,
                0.0,
                -5208520.35574006,
                -5208613.86694342,
                0.0,
                -233.778008402801,
                0.0,
                0.0,
            ]
        ).reshape((2, 6, 1)),
        {"pressure": jnp.zeros((2, 1))}),
    ],
)
def test_compute_stress_two_particles(particles, state_vars, element, target):
    particles.update_natural_coords(element)
    if element.constraints:
        element.apply_boundary_constraints()
    particles.compute_strain(element, 1.0)
    stress = particles.material.compute_stress(None, particles, state_vars)
    assert jnp.allclose(stress, target)
