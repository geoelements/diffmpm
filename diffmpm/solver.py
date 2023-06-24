import functools
from pathlib import Path

import jax.numpy as jnp
from jax import lax
from jax.experimental.host_callback import id_tap
from jax.tree_util import register_pytree_node_class

import diffmpm.writers as writers
from diffmpm.io import Config
from diffmpm.scheme import USF, USL, _schemes


# TODO: Move to __init__.py mostly.
class MPM:
    def __init__(self, filepath):
        self._config = Config(filepath)
        mesh = self._config.parse()
        out_dir = Path(self._config.parsed_config["output"]["folder"]).joinpath(
            self._config.parsed_config["meta"]["title"],
        )

        write_format = self._config.parsed_config["output"]["format"]
        if write_format == "npz":
            writer = writers.NPZWriter()
        else:
            raise ValueError(f"Specified output format not supported: {write_format}")

        if self._config.parsed_config["meta"]["type"] == "MPMExplicit":
            self.solver = MPMExplicit(
                mesh,
                self._config.parsed_config["meta"]["dt"],
                velocity_update=self._config.parsed_config["meta"]["velocity_update"],
                out_steps=self._config.parsed_config["output"]["step_frequency"],
                out_dir=out_dir,
                writer_func=writer.write,
            )
        else:
            raise ValueError("Wrong type of solver specified.")

    def solve(self):
        """Solve the MPM simulation."""
        res = self.solver.solve_jit(
            self._config.parsed_config["meta"]["nsteps"],
            self._config.parsed_config["external_loading"]["gravity"],
        )
        return res


@register_pytree_node_class
class MPMExplicit:
    __particle_props = ("loc", "velocity", "stress", "strain")

    def __init__(
        self,
        mesh,
        dt,
        scheme="usf",
        velocity_update=False,
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
        self.out_steps = out_steps
        self.out_dir = out_dir
        self.writer_func = writer_func
        self.mesh.apply_on_elements("set_particle_element_ids")
        self.mesh.apply_on_elements("compute_volume")
        self.mesh.apply_on_particles(
            "compute_volume", args=(self.mesh.elements.total_elements,)
        )

    def tree_flatten(self):
        children = (self.mesh,)
        aux_data = (
            self.dt,
            self.scheme,
            self.velocity_update,
            self.out_steps,
            self.out_dir,
            self.writer_func,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            *children,
            aux_data[0],
            scheme=aux_data[1],
            velocity_update=aux_data[2],
            out_steps=aux_data[3],
            out_dir=aux_data[4],
            writer_func=aux_data[5],
        )

    def jax_writer(self, func, args):
        id_tap(func, args)

    def solve(self, nsteps: int, gravity: float | jnp.ndarray):
        from collections import defaultdict
        from tqdm import tqdm

        result = defaultdict(list)
        for step in tqdm(range(nsteps)):
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

    def solve_jit(self, nsteps: int, gravity: float | jnp.ndarray):
        def _step(i, data):
            self, nsteps = data
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
                    functools.partial(self.writer_func, out_dir=self.out_dir),
                    (arrays, i),
                )

            lax.cond(
                (i + 1) % self.out_steps == 0, _write, lambda s, i: None, self, i + 1
            )
            return (self, nsteps)

        _, nsteps = lax.fori_loop(0, nsteps, _step, (self, nsteps))
