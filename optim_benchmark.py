from typing import NamedTuple
from functools import partial
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

from diffmpm.constraint import Constraint
from diffmpm.element import Quad4N, Quad4NState
from diffmpm.explicit import ExplicitSolver
from diffmpm.forces import NodalForce
from diffmpm.functions import Unit
from diffmpm.io import Config
from diffmpm.materials import init_simple, init_linear_elastic
from diffmpm.particle import _ParticlesState, init_particle_state

jax.config.update("jax_platform_name", "cpu")

config = Config("./benchmarks/2d/uniaxial_stress/mpm-uniaxial-stress.toml")
# config = Config("./benchmarks/2d/uniaxial_particle_traction/mpm-particle-traction.toml")
# config = Config("./benchmarks/2d/uniaxial_nodal_forces/mpm-nodal-forces.toml")
# config = Config("./benchmarks/2d/hydrostatic_column/mpm.toml")
# parsed_config = config.parse()
# cnf = [NodalForce(node_ids=jnp.array([0, 1]), function=Unit(-1), dir=1, force=1.5)]
# material = NamedTuple("Simple", density=1, E=1, state_vars={})
# ploc = jnp.array([[0.5, 0.5], [0.5, 0.5]]).reshape(2, 1, 2)
# pmat = material(density=1.0, E=1.0, state_vars={})
# pmat = init_simple({"density": 1, "E": 100, "id": 1})
# peids = jnp.array([1])
# particles = [init_particle_state(ploc, pmat, peids)]

# cls = Quad4N(total_elements=1)
# elements = cls.init_state(
#     (1, 1),
#     1,
#     (1, 1),
#     [(jnp.array([0]), Constraint(0, 2))],
#     concentrated_nodal_forces=cnf,
# )

solver = ExplicitSolver(
    el_type=config.parsed_config["elementor"],
    tol=1e-12,
    scheme=config.parsed_config["meta"]["scheme"],
    dt=config.parsed_config["meta"]["dt"],
    velocity_update=config.parsed_config["meta"]["velocity_update"],
    sim_steps=config.parsed_config["meta"]["nsteps"],
    out_steps=config.parsed_config["output"]["step_frequency"],
    out_dir=config.parsed_config["output"]["format"],
    gravity=config.parsed_config["external_loading"]["gravity"],
)

init_vals = solver.init_state(
    {
        "elements": config.parsed_config["elements"],
        "particles": config.parsed_config["particles"],
        "particle_surface_traction": config.parsed_config["particle_surface_traction"],
    }
)

jit_updated = init_vals
jitted_update = jax.jit(solver.update)
for step in tqdm(range(20)):
    # jit_updated = solver.update(jit_updated, step + 1)
    jit_updated = jitted_update(jit_updated, step + 1)

true_vel = jit_updated.particles[0].stress


def compute_loss(params, *, solver, target_vel, config):
    # material = init_simple({"E": params, "density": 1, "id": -1})
    material = init_linear_elastic(
        {"youngs_modulus": params, "density": 1, "poisson_ratio": 0, "id": -1}
    )
    # breakpoint()
    particles_ = [
        init_particle_state(
            config.parsed_config["particles"][0].loc,
            material,
            config.parsed_config["particles"][0].element_ids,
            init_vel=jnp.asarray([1.0, 0.0]),
        )
    ]
    init_vals = solver.init_state(
        {
            "elements": config.parsed_config["elements"],
            "particles": particles_,
            "particle_surface_traction": config.parsed_config[
                "particle_surface_traction"
            ],
        }
    )
    result = init_vals
    for step in tqdm(range(20), leave=False):
        result = jitted_update(result, step + 1)
    vel = result.particles[0].stress
    loss = jnp.linalg.norm(vel - target_vel)
    return loss


def optax_adam(params, niter, mpm, target_vel, config):
    # Initialize parameters of the model + optimizer.
    start_learning_rate = 1
    optimizer = optax.adam(start_learning_rate)
    opt_state = optimizer.init(params)

    param_list = []
    loss_list = []
    # A simple update loop.
    t = tqdm(range(niter), desc=f"E: {params}")
    partial_f = partial(compute_loss, solver=mpm, target_vel=target_vel, config=config)
    for _ in t:
        lo, grads = jax.value_and_grad(partial_f, argnums=0)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        t.set_description(f"YM: {params}")
        param_list.append(params)
        loss_list.append(lo)
    return param_list, loss_list


params = 900.5
# material = init_simple({"E": params, "density": 1, "id": -1})
material = init_linear_elastic(
    {"youngs_modulus": params, "density": 1, "poisson_ratio": 0, "id": -1}
)
particles = [
    init_particle_state(
        config.parsed_config["particles"][0].loc,
        material,
        config.parsed_config["particles"][0].element_ids,
    )
]

init_vals = solver.init_state(
    {
        "elements": config.parsed_config["elements"],
        "particles": particles,
        "particle_surface_traction": config.parsed_config["particle_surface_traction"],
    }
)
param_list, loss_list = optax_adam(
    params, 100, solver, true_vel, config
)  # ADAM optimizer

fig, ax = plt.subplots(1, 2, figsize=(16, 6))
ax[0].plot(param_list, "ko", markersize=2, label="E")
ax[0].grid()
ax[0].legend()
ax[1].plot(loss_list, "ko", markersize=2, label="Loss")
ax[1].grid()
ax[1].legend()
plt.show()
