import jax.numpy as jnp
import matplotlib.pyplot as plt

from diffmpm.constraint import Constraint
from diffmpm.element import Linear1D
from diffmpm.material import SimpleMaterial
from diffmpm.mesh import Mesh1D
from diffmpm.particle import Particles
from diffmpm.solver import MPMExplicit

E = 100
material = SimpleMaterial({"E": E, "density": 1})
cons = [(jnp.array([0]), Constraint(0, 0.0))]
elements = Linear1D(1, 1, 1, cons)
particles = Particles(jnp.array([0.5]).reshape(1, 1, 1), material, jnp.array([0]))
# b1 = jnp.pi * 0.5
# velocity = 0.1 * jnp.sin(b1 * particles.loc)
# particles.velocity = velocity
particles.velocity += 0.1
# particles.set_mass_volume(1.0)
dt = 0.01
nsteps = 1000
mesh = Mesh1D({"particles": [particles], "elements": elements})

mpm = MPMExplicit(mesh, dt, scheme="usf", velocity_update=True)
result = mpm.solve_jit(nsteps, 0)


def analytical_vibration(E, rho, v0, x_loc, L, dt, nsteps):
    tt, vt, xt = [], [], []
    t = 0
    for _ in range(nsteps):
        omega = 1.0 / L * jnp.sqrt(E / rho)
        v = v0 * jnp.cos(omega * t)
        x = x_loc * jnp.exp(v0 / (L * omega) * jnp.sin(omega * t))
        vt.append(v)
        xt.append(x)
        tt.append(t)
        t += dt
    return tt, vt, xt


# analytical solution at the end of the bar
ta, va, xa = analytical_vibration(
    E=E, rho=1, v0=0.1, x_loc=0.5, nsteps=nsteps, dt=dt, L=1.0
)
breakpoint()
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
ax[0].plot(ta, va, "r", label="analytical")
ax[0].plot(ta, result["velocity"].squeeze(), "ob", markersize=2, label="mpm")
ax[0].legend()
ax[0].set_title("Velocity")

ax[1].plot(ta, xa, "r", label="analytical")
ax[1].plot(ta, result["position"].squeeze(), "ob", markersize=2, label="mpm")
ax[1].legend()
ax[1].set_title("Position")

fig, ax = plt.subplots()
ax.plot(
    ta,
    result["strain_energy"].squeeze(),
    "r",
    linewidth=1,
    alpha=0.7,
    label="Strain energy",
)
ax.plot(
    ta,
    result["kinetic_energy"].squeeze(),
    "b",
    linewidth=1,
    alpha=0.7,
    label="Kinetic energy",
)
ax.plot(
    ta,
    result["total_energy"].squeeze(),
    "k",
    linewidth=1,
    alpha=0.7,
    label="Total energy",
)
ax.legend()
ax.set_title("Energies")

plt.show()
