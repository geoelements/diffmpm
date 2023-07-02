import functools
from pathlib import Path

import jax.numpy as jnp
from jax import lax
from jax.experimental.host_callback import id_tap
from jax.tree_util import register_pytree_node_class

from diffmpm.scheme import USF, USL, _schemes


@register_pytree_node_class
class MPMExplicit:
    __particle_props = ("loc", "velocity", "stress", "strain")

    def __init__(
        self,
        mesh,
        dt,
        scheme="usf",
        velocity_update=False,
        sim_steps=1,
        out_steps=1,
        out_dir="results/",
        writer_func=None,
    ):
        if scheme == "usf":
            self.mpm_scheme = USF(mesh, dt, velocity_update)
        elif scheme == "usl":
            self.mpm_scheme = USL(mesh, dt, velocity_update)
        else:
            raise ValueError(f"Please select scheme from {_schemes}. Found {scheme}")
        self.mesh = mesh
        self.dt = dt
        self.scheme = scheme
        self.velocity_update = velocity_update
        self.sim_steps = sim_steps
        self.out_steps = out_steps
        self.out_dir = out_dir
        self.writer_func = writer_func
        self.mpm_scheme.mesh.apply_on_elements("set_particle_element_ids")
        self.mpm_scheme.mesh.apply_on_elements("compute_volume")
        self.mpm_scheme.mesh.apply_on_particles(
            "compute_volume", args=(self.mesh.elements.total_elements,)
        )

    def tree_flatten(self):
        children = (self.mesh,)
        aux_data = {
            "dt": self.dt,
            "scheme": self.scheme,
            "velocity_update": self.velocity_update,
            "sim_steps": self.sim_steps,
            "out_steps": self.out_steps,
            "out_dir": self.out_dir,
            "writer_func": self.writer_func,
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            *children,
            aux_data["dt"],
            scheme=aux_data["scheme"],
            velocity_update=aux_data["velocity_update"],
            sim_steps=aux_data["sim_steps"],
            out_steps=aux_data["out_steps"],
            out_dir=aux_data["out_dir"],
            writer_func=aux_data["writer_func"],
        )

    def jax_writer(self, func, args):
        id_tap(func, args)

    def solve(self, gravity: float | jnp.ndarray):
        from collections import defaultdict

        from tqdm import tqdm

        result = defaultdict(list)
        for step in tqdm(range(self.sim_steps)):
            self.mpm_scheme.compute_nodal_kinematics()
            self.mpm_scheme.precompute_stress_strain()
            self.mpm_scheme.compute_forces(gravity, step)
            self.mpm_scheme.compute_particle_kinematics()
            self.mpm_scheme.postcompute_stress_strain()
            for pset in self.mesh.particles:
                result["position"].append(pset.loc)
                result["velocity"].append(pset.velocity)
                result["stress"].append(pset.stress[:, :2, 0])
                result["strain"].append(pset.strain[:, :2, 0])

        result = {k: jnp.asarray(v) for k, v in result.items()}
        return result

    def solve_jit(self, gravity: float | jnp.ndarray):
        def _step(i, data):
            self = data
            self.mpm_scheme.compute_nodal_kinematics()
            self.mpm_scheme.precompute_stress_strain()
            self.mpm_scheme.compute_forces(gravity, i)
            self.mpm_scheme.compute_particle_kinematics()
            self.mpm_scheme.postcompute_stress_strain()

            def _write(self, i):
                arrays = {}
                for name in self.__particle_props:
                    arrays[name] = jnp.array(
                        [
                            getattr(self.mesh.particles[j], name).squeeze()
                            for j in range(len(self.mesh.particles))
                        ]
                    )
                self.jax_writer(
                    functools.partial(
                        self.writer_func, out_dir=self.out_dir, max_steps=self.sim_steps
                    ),
                    (arrays, i),
                )

            if self.writer_func is not None:
                lax.cond(
                    i % self.out_steps == 0,
                    _write,
                    lambda s, i: None,
                    self,
                    i,
                )
            return self

        self = lax.fori_loop(0, self.sim_steps, _step, self)
        arrays = {}
        for name in self.__particle_props:
            arrays[name] = jnp.array(
                [
                    getattr(self.mesh.particles[j], name)
                    for j in range(len(self.mesh.particles))
                ]
            ).squeeze()
        return arrays
