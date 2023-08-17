import jax.numpy as jnp
import pytest

from diffmpm.constraint import Constraint
from diffmpm.element import Quadrilateral4Node
from diffmpm.forces import NodalForce
from diffmpm.functions import Unit
from diffmpm.materials import init_simple
from diffmpm.particle import init_particle_state
import diffmpm.particle as dpar


class TestLinear1D:
    pass


class TestQuadrilateral4Node:
    @pytest.fixture
    def elements(self):
        return Quadrilateral4Node((1, 1), 1, (1.0, 1.0), [])

    @pytest.fixture
    def particles(self):
        loc = jnp.array([[0.5, 0.5], [0.5, 0.5]]).reshape(2, 1, 2)
        material = init_simple({"id": 0, "E": 1, "density": 1})
        return init_particle_state(loc, material, jnp.array([0, 0]))

    @pytest.mark.parametrize(
        "particle_coords, expected",
        [
            (
                jnp.array([[0, 0]]).reshape(-1, 1, 2),
                jnp.array([0.25, 0.25, 0.25, 0.25]).reshape(1, 4, 1, 1),
            ),
            (
                jnp.array([[-1, -1]]).reshape(-1, 1, 2),
                jnp.array([1, 0, 0, 0]).reshape(1, 4, 1, 1),
            ),
            (
                jnp.array([[1, 1]]).reshape(-1, 1, 2),
                jnp.array([0, 0, 1, 0]).reshape(1, 4, 1, 1),
            ),
        ],
    )
    def test_shapefn(self, elements, particle_coords, expected):
        sfn = elements.shapefn(particle_coords)
        assert jnp.allclose(sfn, expected)

    @pytest.mark.parametrize(
        "particle_coords, expected",
        [
            (
                jnp.array([[0, 0]]).reshape(-1, 1, 2),
                jnp.array(
                    [[-0.25, -0.25], [0.25, -0.25], [0.25, 0.25], [-0.25, 0.25]]
                ).reshape(-1, 4, 2),
            ),
            (
                jnp.array([[-1, -1]]).reshape(-1, 1, 2),
                jnp.array([[-0.5, -0.5], [0.5, 0.0], [0.0, 0.0], [0.0, 0.5]]).reshape(
                    -1, 4, 2
                ),
            ),
            (
                jnp.array([[1, 1]]).reshape(-1, 1, 2),
                jnp.array([[0.0, 0.0], [0.0, -0.5], [0.5, 0.5], [-0.5, 0.0]]).reshape(
                    -1, 4, 2
                ),
            ),
        ],
    )
    def test_shapefn_natural_grad(self, elements, particle_coords, expected):
        sfn_grad = elements._shapefn_natural_grad(particle_coords)
        assert jnp.allclose(sfn_grad, expected)

    @pytest.mark.parametrize(
        "particle_coords",
        [
            jnp.array([[0.5, 0.5]]).reshape(-1, 1, 2),
            jnp.array([[-0.5, -0.5]]).reshape(-1, 1, 2),
        ],
    )
    def test_shapefn_grad(self, elements, particle_coords):
        nodal_coords = jnp.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        ).reshape(-1, 4, 2)
        sfn_natural_grad = elements._shapefn_natural_grad(particle_coords) * 2
        sfn_grad = elements.shapefn_grad(particle_coords, nodal_coords)
        assert jnp.allclose(sfn_grad, sfn_natural_grad)

    def test_element_node_ids(self, elements):
        node_ids = elements.id_to_node_ids(0).reshape(
            -1,
        )
        assert jnp.all(node_ids == jnp.array([0, 1, 3, 2]))

    def test_element_node_loc(self, elements):
        node_loc = elements.id_to_node_loc(0)
        true_loc = jnp.array([[0.0, 0.0], [1, 0], [1, 1], [0, 1]]).reshape(4, 1, 2)
        assert jnp.all(node_loc == true_loc)

    def test_element_node_vel(self, elements):
        elements.nodes = elements.nodes.replace(
            velocity=elements.nodes.velocity + jnp.array([1, 1])
        )
        node_vel = elements.id_to_node_vel(0)
        true_vel = jnp.array([[1.0, 1.0], [1, 1], [1, 1], [1, 1]]).reshape(4, 1, 2)
        assert jnp.all(node_vel == true_vel)

    def test_compute_nodal_mass(self, elements, particles):
        particles = particles.replace(mass=particles.mass + 1)
        nodal_mass, _ = elements.compute_nodal_mass(particles)
        true_mass = jnp.ones((4, 1, 1))
        assert jnp.all(nodal_mass == true_mass)

    def test_compute_nodal_momentum(self, elements, particles):
        particles = particles.replace(velocity=particles.velocity + 1)
        nodal_momentum, _ = elements.compute_nodal_momentum(particles)
        true_momentum = jnp.ones((4, 1, 1)) * 0.5
        assert jnp.all(nodal_momentum == true_momentum)

    def test_compute_external_force(self, elements, particles):
        particles = particles.replace(f_ext=particles.f_ext + 1)
        nodal_f_ext, _ = elements.compute_external_force(particles)
        true_fext = jnp.ones((4, 1, 1)) * 0.5
        assert jnp.all(nodal_f_ext == true_fext)

    @pytest.mark.parametrize(
        "gravity, expected",
        [
            (0, jnp.zeros((4, 1, 2))),
            (
                jnp.array([10, 0]),
                jnp.array([[10, 0], [10, 0], [10, 0], [10, 0]]).reshape(4, 1, 2),
            ),
        ],
    )
    def test_compute_body_force(self, elements, particles, gravity, expected):
        particles = particles.replace(mass=particles.mass + 1)
        nodal_f_ext, _ = elements.compute_body_force(particles, gravity)
        assert jnp.all(nodal_f_ext == expected)

    def test_apply_concentrated_nodal_force(self, particles):
        cnf_1 = NodalForce(
            node_ids=jnp.array([0, 2]),
            function=Unit(-1),
            dir=0,
            force=1,
        )
        cnf_2 = NodalForce(
            node_ids=jnp.array([2]),
            function=Unit(-1),
            dir=1,
            force=1,
        )
        elements = Quadrilateral4Node(
            (1, 1), 1, 1, [], concentrated_nodal_forces=[cnf_1, cnf_2]
        )
        elements.nodes = elements.nodes.replace(f_ext=elements.nodes.f_ext + 2)
        nodal_f_ext, _ = elements.apply_concentrated_nodal_forces(particles, 1)
        assert jnp.all(
            nodal_f_ext == jnp.array([[3, 2], [2, 2], [3, 3], [2, 2]]).reshape(4, 1, 2)
        )

    def test_apply_boundary_constraints(self):
        cons = [
            (jnp.array([0, 1]), Constraint(0, 0)),
            (jnp.array([0]), Constraint(1, 2)),
        ]
        elements = Quadrilateral4Node((1, 1), 1, (1.0, 1.0), cons)
        elements.nodes = elements.nodes.replace(velocity=elements.nodes.velocity + 1)
        node_state = elements.apply_boundary_constraints()
        assert jnp.all(
            node_state.velocity
            == jnp.array([[0, 2], [0, 1], [1, 1], [1, 1]]).reshape(4, 1, 2)
        )

    def test_update_nodal_acceleration_velocity(self, elements, particles):
        f_ext = elements.nodes.f_ext + jnp.array([1, 0])
        mass = elements.nodes.mass.at[:].set(2)
        elements.nodes = elements.nodes.replace(mass=mass, f_ext=f_ext)
        nodal_acc, _ = elements.update_nodal_acceleration(particles, 0.1)
        assert jnp.allclose(
            nodal_acc,
            jnp.array([[0.5, 0.0], [0.5, 0], [0.5, 0], [0.5, 0]]),
        )
        nodal_vel, _ = elements.update_nodal_velocity(particles, 0.1)
        assert jnp.allclose(
            nodal_vel,
            jnp.array([[0.05, 0.0], [0.05, 0], [0.05, 0], [0.05, 0]]),
        )
        elements.nodes = elements.nodes.replace(velocity=nodal_vel)
        nodal_mom, _ = elements.update_nodal_momentum(particles, 0.1)
        assert jnp.allclose(
            nodal_mom,
            jnp.array([[0.1, 0.0], [0.1, 0], [0.1, 0], [0.1, 0]]),
        )

    def test_set_particle_element_ids(self, elements, particles):
        particles = particles.replace(element_ids=jnp.array([-1, -1]))
        particles = elements.set_particle_element_ids(particles)
        assert jnp.all(particles.element_ids == jnp.array([0, 0]))

    def test_compute_internal_force(self, elements, particles):
        particles = dpar.compute_volume(particles, elements, 1)
        particles = particles.replace(stress=particles.stress + 1)
        nodal_f_int, _ = elements.compute_internal_force(particles)
        assert jnp.allclose(
            nodal_f_int,
            jnp.array([[1, 1], [0, 0], [0, 0], [-1, -1]]).reshape(4, 1, 2),
        )

    def test_compute_volume(self, elements):
        elements.compute_volume()
        assert jnp.allclose(elements.volume, jnp.array([1]).reshape(1, 1, 1))

    def test_apply_particle_traction_forces(self, elements, particles):
        particles = particles.replace(traction=particles.traction + jnp.array([1, 0]))
        nodal_f_ext, _ = elements.apply_particle_traction_forces(particles)
        assert jnp.allclose(
            nodal_f_ext,
            jnp.array([[0.5, 0], [0.5, 0], [0.5, 0], [0.5, 0]]).reshape(4, 1, 2),
        )
