import jax.numpy as jnp
import matplotlib.pyplot as plt
from diffmpm.material import Material
from diffmpm.particle import Particles
from diffmpm.mesh import Mesh1D, Mesh2D


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
    assert jnp.allclose(mesh.nodes.momentum, jnp.array([0.25, 1.0, 0.75]))


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
        jnp.array([0.35, 0.85, 1.4125, 1.9375]),
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


def test_node_mass_par_mass_update():
    material = Material(2, 1)
    mesh = Mesh1D(2, material, 2, jnp.array([0]), ppe=2)
    mesh.particles.mass = jnp.ones_like(mesh.particles.x)
    mesh._update_node_mass_par_mass()
    assert jnp.allclose(mesh.nodes.mass, jnp.array([0.5, 2, 1.5]))


def test_node_fext_par_mass():
    material = Material(2, 1)
    mesh = Mesh1D(2, material, 2, jnp.array([0]), ppe=2)
    mesh.particles.mass = jnp.ones_like(mesh.particles.x)
    mesh._update_node_fext_par_mass(10)
    assert jnp.allclose(mesh.nodes.f_ext, jnp.array([5, 20, 15]))


def test_node_fint_par_mass():
    material = Material(2, 1)
    mesh = Mesh1D(2, material, 2, jnp.array([0]), ppe=2)
    mesh.particles.mass = jnp.ones_like(mesh.particles.x)
    mesh.particles.stress = jnp.ones_like(mesh.particles.x) * 2
    mesh._update_node_fint_par_mass()
    assert jnp.allclose(mesh.nodes.f_int, jnp.array([4, 0, -4]))


def test_node_fext_par_fext():
    material = Material(2, 1)
    mesh = Mesh1D(2, material, 2, jnp.array([0]), ppe=2)
    mesh.particles.f_ext = jnp.ones_like(mesh.particles.x)
    mesh._update_node_fext_par_fext()
    assert jnp.allclose(mesh.nodes.f_ext, jnp.array([0.5, 2, 1.5]))


def test_mesh_solve():
    material = Material(100, 1)
    mesh = Mesh1D(1, material, 1, jnp.array([0]), ppe=1)
    b1 = jnp.pi * 0.5 / mesh.domain_size
    velocity = 0.1 * jnp.sin(b1 * mesh.particles.x)
    mesh.set_particle_velocity(velocity)
    result = mesh.solve(dt=0.1, nsteps=5, mpm_scheme="USF")
    assert jnp.allclose(
        jnp.array(result["position"]).squeeze(),
        jnp.array(
            [
                0.49641446609406725,
                0.48931928464800056,
                0.48593443288575755,
                0.48959197190820614,
                0.4966373581704467,
            ]
        ),
    )


def test_analytical_solve():
    E = 4 * jnp.pi**2
    material = Material(E, 1)
    mesh = Mesh1D(3, material, 1, jnp.array([0]), ppe=2)
    mesh2 = Mesh1D(3, material, 1, jnp.array([0]), ppe=2)
    # velocity = jnp.array([0.1])
    velocity = 0.1 * jnp.sin(mesh.particles.x)
    mesh.set_particle_velocity(velocity)
    mesh2.set_particle_velocity(velocity)
    dt = 0.01
    duration = 1
    nsteps = int(duration / dt)
    r = mesh2.solve(dt=dt, nsteps=nsteps, mpm_scheme="USL")
    result = mesh.solve_jit(dt=dt, nsteps=nsteps, mpm_scheme="USL")

    def analytical_vibration(E, rho, v0, x_loc, duration, dt, L):
        t = jnp.arange(0, duration, dt)
        omega = 1 / L * jnp.sqrt(E / rho)
        v = v0 * jnp.cos(omega * t)
        x = x_loc * jnp.exp(v0 / (L * omega) * jnp.sin(omega * t))
        return x, v, t

    x1, v1, t = analytical_vibration(E, 1, 0.1, 0.25, duration, dt, 1)
    x2, v2, t = analytical_vibration(E, 1, 0.1, 0.5, duration, dt, 1)
    # fig, ax = plt.subplots()
    # ax.plot(t, v1, "r--", linewidth=1, markersize=5, label="analytical")
    # ax.plot(t, v1, "r+-", linewidth=1, markersize=5, label="analytical")
    # ax.plot(
    #     t,
    #     result["velocity"],
    #     "k+--",
    #     linewidth=1,
    #     markersize=6,
    #     label="diffmpm-jit",
    # )
    # ax.plot(
    #     t,
    #     r["velocity"],
    #     "gx--",
    #     linewidth=1,
    #     markersize=4,
    #     label="diffmpm",
    # )
    # ax.grid()
    # ax.legend()
    # ax.set_xlabel("Time (s)")
    # ax.set_ylabel("Velocity (m/s)")
    # plt.show()
    # assert jnp.allclose(result["velocity"], v, rtol=1e-3, atol=1e-4)


class TestMesh2D:
    material = Material(1, 1)
    particles = Particles(
        1,
        jnp.array([[0.5, 0.5], [1.5, 0.5]]),
        jnp.array([[0, 0], [0, 0]]),
        1,
        jnp.array([0, 1]),
        jnp.array([[0.1, 0.1], [0.1, 0.1]]),
        0,
        0,
        0,
        0,
        0,
        nelements=2,
    )
    mesh = Mesh2D(
        (2, 1), material, (0, 2, 0, 1), jnp.array([0]), particles=particles
    )

    def test_2d_get_element_node_ids(self):
        material = Material(100, 1)
        mesh = Mesh2D((4, 3), material, (0, 1, 0, 1), jnp.array([0]), ppe=1)
        element_id = 10
        node_ids = mesh._get_element_node_ids(element_id)
        correct_nodes = jnp.array([12, 13, 18, 17])
        assert jnp.all(node_ids == correct_nodes)

    def test_2d_get_element_node_pos(self):
        material = Material(100, 1)
        mesh = Mesh2D((4, 3), material, (0, 4, 0, 3), jnp.array([0]), ppe=1)
        element_id = 10
        node_pos = mesh._get_element_node_pos(element_id)
        correct_nodes_pos = jnp.array(
            [[2.0, 2.0], [3.0, 2.0], [3.0, 3.0], [2.0, 3.0]]
        )
        assert jnp.all(node_pos == correct_nodes_pos)

    def test_2d_get_element_node_vel(self):
        material = Material(100, 1)
        mesh = Mesh2D((4, 3), material, (0, 4, 0, 3), jnp.array([0]), ppe=1)
        mesh.nodes.velocity = jnp.array(
            [jnp.arange(mesh.nodes.nnodes), jnp.arange(mesh.nodes.nnodes)]
        ).T
        element_id = 10
        node_vel = mesh._get_element_node_vel(element_id)
        correct_nodes_vel = jnp.array([[12, 12], [13, 13], [18, 18], [17, 17]])
        assert jnp.all(node_vel == correct_nodes_vel)

    def test_2d_particle_element_mapping_update(self):
        material = Material(1, 1)
        particles = Particles(
            1,
            jnp.array([[2.5, 2.5], [0.5, 1.5]]),
            jnp.array([[0, 0]]),
            1,
            jnp.array([0, 0]),
            0,
            0,
            0,
            0,
            0,
            0,
            nelements=9,
        )
        mesh = Mesh2D(
            (4, 3), material, (0, 4, 0, 3), jnp.array([0]), particles=particles
        )
        mesh._update_particle_element_ids()
        assert jnp.all(mesh.particles.element_ids == jnp.array([10, 4]))
        mesh.particles.x = jnp.array([[1.5, 0.5], [0.5, 0.5]])
        mesh._update_particle_element_ids()
        assert jnp.all(mesh.particles.element_ids == jnp.array([1, 0]))
        mesh.particles.x = jnp.array([[0.5, 2.5], [3.5, 2.5]])
        mesh._update_particle_element_ids()
        assert jnp.all(mesh.particles.element_ids == jnp.array([8, 11]))

    def test_2d_particle_xi_update(self):
        material = Material(1, 1)
        particles = Particles(
            1,
            jnp.array([[1.5, 1.5], [0.75, 1.5]]),
            jnp.array([[0, 0]]),
            1,
            jnp.array([0, 0]),
            0,
            0,
            0,
            0,
            0,
            0,
            nelements=9,
        )
        mesh = Mesh2D(
            (4, 3), material, (0, 4, 0, 3), jnp.array([0]), particles=particles
        )
        mesh._update_particle_element_ids()
        mesh._update_particle_natural_coords()
        assert jnp.allclose(mesh.particles.xi, jnp.array([[0, 0], [0.5, 0]]))
        mesh.particles.x += 1
        mesh._update_particle_element_ids()
        mesh._update_particle_natural_coords()
        assert jnp.allclose(mesh.particles.xi, jnp.array([[0, 0], [0.5, 0]]))
        mesh.particles.x -= 0.5
        mesh._update_particle_element_ids()
        mesh._update_particle_natural_coords()
        assert jnp.allclose(
            mesh.particles.xi, jnp.array([[-1, -1], [-0.5, -1]])
        )

    def test_2d_update_particle_strain(self):
        material = Material(1, 1)
        particles = Particles(
            1,
            jnp.array([[0.5, 0.5], [1.5, 0.5]]),
            jnp.array([[0, 1]]),
            1,
            jnp.array([0, 1]),
            jnp.array([[0.1, 0.1], [0.1, 0.1]]),
            0,
            0,
            0,
            0,
            0,
            nelements=9,
        )
        mesh = Mesh2D(
            (2, 1), material, (0, 2, 0, 1), jnp.array([0]), particles=particles
        )
        mesh.nodes.velocity = jnp.array(
            [[0, 0], [1, 1], [1, 1], [1, 1], [2, 2], [1, 1]]
        )
        mesh._update_particle_natural_coords()
        mesh._update_particle_strain(0.1)
        assert jnp.allclose(
            mesh.particles.strain, jnp.array([[0.1, 0.1], [0.1, 0.1]])
        )

    def test_2d_update_nodes_acc_vel(self):
        material = Material(1, 1)
        particles = Particles(
            1,
            jnp.array([[0.5, 0.5], [1.5, 0.5]]),
            jnp.array([[0, 1]]),
            1,
            jnp.array([0, 1]),
            jnp.array([[0.1, 0.1], [0.1, 0.1]]),
            0,
            0,
            0,
            0,
            0,
            nelements=9,
        )
        mesh = Mesh2D(
            (2, 1), material, (0, 2, 0, 1), jnp.array([0]), particles=particles
        )
        mesh.nodes.f_int = jnp.array(
            [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]]
        )
        mesh._update_nodes_acc_vel(0.1)
        assert jnp.allclose(
            mesh.nodes.velocity, jnp.zeros_like(mesh.nodes.velocity)
        )
        mesh.nodes.mass += 1
        mesh._update_nodes_acc_vel(0.1)
        assert jnp.allclose(
            mesh.nodes.velocity,
            jnp.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]]) * 0.1,
        )

    def test_2d_update_nodes_mom_vel(self):
        material = Material(1, 1)
        particles = Particles(
            1,
            jnp.array([[0.5, 0.5], [1.5, 0.5]]),
            jnp.array([[0, 1]]),
            1,
            jnp.array([0, 1]),
            jnp.array([[0.1, 0.1], [0.1, 0.1]]),
            0,
            0,
            0,
            0,
            0,
            nelements=9,
        )
        mesh = Mesh2D(
            (2, 1), material, (0, 2, 0, 1), jnp.array([0]), particles=particles
        )
        mesh.nodes.momentum = jnp.array(
            [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]], dtype=jnp.float32
        )
        mesh._update_nodes_mom_vel()
        assert jnp.allclose(
            mesh.nodes.velocity, jnp.zeros_like(mesh.nodes.velocity)
        )
        mesh.nodes.mass += 2
        mesh._update_nodes_mom_vel()
        assert jnp.allclose(
            mesh.nodes.velocity,
            jnp.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]]) * 0.5,
        )

    def test_2d_update_node_momentum_force(self):
        self.mesh.nodes.f_int += 1
        self.mesh.nodes.f_ext += 2
        self.mesh.nodes.f_damp += 3
        self.mesh._update_node_momentum_force(0.1)
        assert jnp.allclose(
            self.mesh.nodes.momentum,
            jnp.ones_like(self.mesh.nodes.momentum) * 0.6,
        )

    def test_update_node_momentum_par_vel(self):
        self.mesh.particles.velocity = self.mesh.particles.velocity.at[:].set(
            jnp.array([2, 1])
        )
        self.mesh._update_node_momentum_par_vel()
        assert jnp.allclose(
            self.mesh.nodes.momentum,
            jnp.array(
                [
                    [0.5, 0.25],
                    [1, 0.5],
                    [0.5, 0.25],
                    [0.5, 0.25],
                    [1, 0.5],
                    [0.5, 0.25],
                ]
            ),
        )

    def test_transfer_node_force_vel_par(self):
        self.mesh.nodes.f_int += 2
        self.mesh.nodes.mass += 1
        self.mesh._transfer_node_force_vel_par(0.1)
        assert jnp.allclose(
            self.mesh.particles.velocity, jnp.array([[0.3, 0.3], [0.3, 0.3]])
        )
