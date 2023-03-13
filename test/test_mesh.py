import jax.numpy as jnp
from diffmpm.material import Material
from diffmpm.mesh import Mesh1D


def test_particles_uniform_initialization():
    material = Material(1, 1)
    mesh = Mesh1D(
        2, material, 2, jnp.array([0]), ppe=3, particle_distribution="uniform"
    )
    assert jnp.allclose(
        mesh.particles.x, jnp.array([1 / 6, 1 / 2, 5 / 6, 7 / 6, 3 / 2, 11 / 6])
    )


def test_particle_initial_element_mapping():
    material = Material(1, 1)
    mesh = Mesh1D(2, material, 2, jnp.array([0]), ppe=3)
    assert (mesh.particles.element_ids == jnp.array([0, 0, 0, 1, 1, 1])).all()


def test_particle_element_mapping_update():
    material = Material(1, 1)
    mesh = Mesh1D(2, material, 2, jnp.array([0]), ppe=3)
    mesh.particles.x += 1
    mesh._update_particle_element_ids()
    assert (
        mesh.particles.element_ids == jnp.array([1, 1, 1, -1, -1, -1])
    ).all()
    mesh.particles.x -= 2
    mesh._update_particle_element_ids()
    assert (
        mesh.particles.element_ids == jnp.array([-1, -1, -1, 0, 0, 0])
    ).all()
    mesh.particles.x -= 1
    mesh._update_particle_element_ids()
    assert (
        mesh.particles.element_ids == jnp.array([-1, -1, -1, -1, -1, -1])
    ).all()
    mesh.particles.x += 5
    mesh._update_particle_element_ids()
    assert (
        mesh.particles.element_ids == jnp.array([-1, -1, -1, -1, -1, -1])
    ).all()


def test_particle_xi_update():
    material = Material(1, 1)
    mesh = Mesh1D(2, material, 2, jnp.array([0]), ppe=3)
    mesh._update_particle_natural_coords()
    assert jnp.allclose(
        mesh.particles.xi, jnp.tile(jnp.array([-2 / 3, 0, 2 / 3]), 2)
    )
    mesh.particles.x += 1
    mesh._update_particle_element_ids()
    mesh._update_particle_natural_coords()
    assert jnp.allclose(
        mesh.particles.xi, jnp.array([-2 / 3, 0, 2 / 3, 0, 0, 0])
    )
    mesh.particles.x -= 2
    mesh._update_particle_element_ids()
    mesh._update_particle_natural_coords()
    assert jnp.allclose(
        mesh.particles.xi, jnp.array([0, 0, 0, -2 / 3, 0, 2 / 3])
    )


def test_element_node_mapping():
    material = Material(1, 1)
    mesh = Mesh1D(2, material, 2, jnp.array([0]), ppe=3)
    assert (
        mesh._get_element_node_pos(0)
        == mesh.nodes.position[jnp.asarray([0, 1])]
    ).all()


def test_particle_strain_stress_update():
    material = Material(2, 1)
    mesh = Mesh1D(2, material, 2, jnp.array([0]), ppe=2)
    mesh.nodes.velocity = jnp.array([0, 1, 1])
    mesh._update_particle_strain(0.1)
    mesh._update_particle_stress()
    # Actual solution:
    # particles.x = [0.25, 0.75, 1.25, 1.75]
    # nodes.v = [0, 1, 1]
    # shapefn_grad = [[-1, 1], [-1, 1], [-1, 1], [-1, 1]]
    # strain_rate[i] = shapefn_grad[i][0] * nodes.v[0] + shapefn_grad[i][1] * nodes.v[1]
    # strain[i] = strain_rate[i] * dt
    assert jnp.allclose(mesh.particles.strain, jnp.array([0.1, 0.1, 0, 0]))
    assert jnp.allclose(mesh.particles.stress, jnp.array([0.2, 0.2, 0, 0]))


def test_nodes_acc_velocity():
    material = Material(2, 1)
    mesh = Mesh1D(2, material, 2, jnp.array([0]), ppe=2)
    mesh.nodes.mass = jnp.ones_like(mesh.nodes.position) * 2
    mesh.nodes.f_int = jnp.ones_like(mesh.nodes.position)
    mesh.nodes.f_ext = jnp.ones_like(mesh.nodes.position)
    mesh._update_nodes_acc_vel(0.1)
    assert jnp.allclose(mesh.nodes.velocity, jnp.array([0.1, 0.1, 0.1]))

    # This shouldn't change velocity since mass is 0
    mesh.nodes.mass = jnp.zeros_like(mesh.nodes.position)
    mesh._update_nodes_acc_vel(0.1)
    assert jnp.allclose(mesh.nodes.velocity, jnp.array([0.1, 0.1, 0.1]))


def test_nodes_mom_velocity():
    material = Material(2, 1)
    mesh = Mesh1D(2, material, 2, jnp.array([0]), ppe=2)
    mesh.nodes.mass = jnp.ones_like(mesh.nodes.position) * 2
    mesh.nodes.momentum = jnp.ones_like(mesh.nodes.position)
    mesh._update_nodes_mom_vel()
    assert jnp.allclose(mesh.nodes.velocity, jnp.array([0.5, 0.5, 0.5]))

    # This should velocity to 0 since mass is 0
    mesh.nodes.mass = jnp.zeros_like(mesh.nodes.position)
    mesh._update_nodes_mom_vel()
    assert jnp.allclose(mesh.nodes.velocity, jnp.array([0.0, 0.0, 0.0]))


def test_nodes_bc_force_update():
    material = Material(2, 1)
    mesh = Mesh1D(2, material, 2, jnp.array([0]), ppe=2)
    mesh.nodes.f_int = jnp.ones_like(mesh.nodes.position)
    mesh.nodes.f_ext = jnp.ones_like(mesh.nodes.position)
    mesh.nodes.f_damp = jnp.ones_like(mesh.nodes.position)
    mesh._update_nodes_bc_force()
    assert jnp.all(mesh.nodes.f_int == jnp.array([0, 1, 1]))
    assert jnp.all(mesh.nodes.f_ext == jnp.array([0, 1, 1]))
    assert jnp.all(mesh.nodes.f_damp == jnp.array([0, 1, 1]))


def test_nodes_bc_mom_vel_update():
    material = Material(2, 1)
    mesh = Mesh1D(2, material, 2, jnp.array([0]), ppe=2)
    mesh.nodes.mass = jnp.ones_like(mesh.nodes.position) * 2
    mesh.nodes.momentum = jnp.ones_like(mesh.nodes.position)
    mesh.nodes.velocity = jnp.ones_like(mesh.nodes.position)
    mesh._update_nodes_bc_mom_vel()
    assert jnp.all(mesh.nodes.momentum == jnp.array([0, 1, 1]))
    assert jnp.all(mesh.nodes.momentum == jnp.array([0, 1, 1]))


def test_nodes_momentum_vel_update():
    material = Material(2, 1)
    mesh = Mesh1D(2, material, 2, jnp.array([0]), ppe=2)
    mesh.particles.velocity = jnp.ones_like(mesh.particles.x)
    mesh._update_node_momentum_par_vel()
    assert jnp.allclose(mesh.nodes.momentum, jnp.array([0.25, 0.5, 1.25]))


def test_transfer_node_force_vel_par():
    material = Material(2, 1)
    mesh = Mesh1D(2, material, 2, jnp.array([0]), ppe=2)
    mesh.nodes.f_int = jnp.ones_like(mesh.nodes.position)
    mesh.nodes.mass = jnp.ones_like(mesh.nodes.position)
    mesh._transfer_node_force_vel_par(0.1)
    assert jnp.allclose(
        mesh.particles.velocity, jnp.array([0.1, 0.1, 0.1, 0.1])
    )


def test_par_pos_node_mom_update():
    material = Material(2, 1)
    mesh = Mesh1D(2, material, 2, jnp.array([0]), ppe=2)
    mesh.nodes.momentum = jnp.array([1, 1, 2])
    mesh.nodes.mass = jnp.ones_like(mesh.nodes.position)
    mesh._update_par_pos_node_mom(0.1)
    assert jnp.allclose(
        mesh.particles.x,
        jnp.array([0.35, 0.85, 1.4625, 1.9875]),
    )


def test_par_pos_vel_node_vel():
    material = Material(2, 1)
    mesh = Mesh1D(2, material, 2, jnp.array([0]), ppe=2)
    mesh.nodes.velocity = jnp.ones_like(mesh.nodes.position)
    mesh._update_par_pos_vel_node_vel(0.1)
    assert jnp.allclose(mesh.particles.velocity, jnp.array([1, 1, 1, 1]))
    assert jnp.allclose(mesh.particles.x, jnp.array([0.35, 0.85, 1.35, 1.85]))


def test_par_vol_density():
    material = Material(2, 1)
    mesh = Mesh1D(2, material, 2, jnp.array([0]), ppe=2)
    mesh.particles.dstrain = jnp.ones_like(mesh.particles.x) * 0.5
    mesh.particles.volume = jnp.ones_like(mesh.particles.x)
    mesh._update_par_vol_density()
    assert jnp.allclose(mesh.particles.volume, jnp.array([1.5, 1.5, 1.5, 1.5]))
    assert jnp.allclose(
        mesh.particles.density, jnp.array([2 / 3, 2 / 3, 2 / 3, 2 / 3])
    )