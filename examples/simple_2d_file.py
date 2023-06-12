import sys
import jax.numpy as jnp
import matplotlib.pyplot as plt
from diffmpm.solver import MPM

mpm = MPM(sys.argv[1])
result = mpm.solve()
# breakpoint()
exit()


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


E = mpm._config._fileconfig["materials"][0]["E"]
nsteps = mpm._config._fileconfig["meta"]["nsteps"]
dt = mpm._config._fileconfig["meta"]["dt"]

# analytical solution at the end of the bar
ta, va, xa = analytical_vibration(
    E=E, rho=1, v0=0.1, x_loc=0.5, nsteps=nsteps, dt=dt, L=1.0
)
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
