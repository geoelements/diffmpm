import jax.numpy as jnp
from jax import grad, jit
import optax
import matplotlib.pyplot as plt

from tqdm import tqdm
from diffmpm.mesh import Mesh1D
from diffmpm.material import Material

E_true = 100
true_mesh = Mesh1D(13, Material(E_true, 1), 25, jnp.array([0]))
init_vel = jnp.ones(len(true_mesh.particles.x)) * 0.1
true_mesh.set_particle_velocity(init_vel)
dt = 0.02
nsteps = 2500
duration = int(dt * nsteps)
target = true_mesh.solve_jit(nsteps=nsteps, dt=dt, mpm_scheme="USF")
target_vel = target["velocity"][6]


@jit
def compute_loss(E, mesh, target_vel):
    mesh.material = Material(E, 1)
    init_vel = jnp.ones(len(mesh.particles.x)) * 0.1
    mesh.set_particle_velocity(init_vel)
    dt = 0.02
    nsteps = 2500
    result = mesh.solve_jit(nsteps=nsteps, dt=dt)
    vel = result["velocity"][6]
    return jnp.linalg.norm(vel - target_vel)


def optax_adam(params, niter, mesh, target_vel):
    # Initialize parameters of the model + optimizer.
    start_learning_rate = 1e-1
    optimizer = optax.adam(start_learning_rate)
    opt_state = optimizer.init(params)

    param_list = []
    # A simple update loop.
    t = tqdm(range(niter), desc=f"E: {params}")
    for _ in t:
        grads = grad(compute_loss)(params, mesh, target_vel)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        t.set_description(f"E: {params}")
        param_list.append(params)
    return param_list


params = 95.0
mesh = Mesh1D(13, Material(params, 1), 25, jnp.array([0]))
result = optax_adam(params, 4, mesh, target_vel)  # ADAM optimizer
# print("E: {}".format(result))

fig, ax = plt.subplots()
ax.plot(result, "ko", markersize=4, label="E")
ax.grid()
ax.legend()
plt.show()
