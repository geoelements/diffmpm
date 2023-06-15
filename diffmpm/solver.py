import jax.numpy as jnp
from jax import lax
from jax.tree_util import register_pytree_node_class
from tqdm import tqdm
from collections import defaultdict

from diffmpm.io import Config
from diffmpm.scheme import USF, USL, _schemes


# TODO: Move to __init__.py mostly.
class MPM:
    def __init__(self, filepath):
        self._config = Config(filepath)
        mesh = self._config.parse()
        if self._config.parsed_config["meta"]["type"] == "MPMExplicit":
            self.solver = MPMExplicit(
                mesh,
                self._config.parsed_config["meta"]["dt"],
                velocity_update=self._config.parsed_config["meta"]["velocity_update"],
            )
        else:
            raise ValueError("Wrong type of solver specified.")

    def solve(self):
        res = self.solver.solve_jit(
            self._config.parsed_config["meta"]["nsteps"],
            self._config.parsed_config["external_loading"]["gravity"],
        )
        return res


@register_pytree_node_class
class MPMExplicit:
    def __init__(self, mesh, dt, scheme="usf", velocity_update=False):
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
        self.mesh.apply_on_elements("set_particle_element_ids")
        self.mesh.apply_on_elements("compute_volume")
        self.mesh.apply_on_particles(
            "compute_volume", args=(self.mesh.elements.total_elements,)
        )

    def tree_flatten(self):
        children = (self.mesh,)
        aux_data = (self.dt, self.scheme, self.velocity_update)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            *children, aux_data[0], scheme=aux_data[1], velocity_update=aux_data[2]
        )

    def solve(self, nsteps: int, gravity: float | jnp.ndarray):
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
        nparticles = sum(pset.loc.shape[0] for pset in self.mesh.particles)
        result = {
            "position": jnp.zeros((nsteps, nparticles, 2)),
            "velocity": jnp.zeros((nsteps, nparticles, 2)),
            "stress": jnp.zeros((nsteps, nparticles, 6)),
            "strain": jnp.zeros((nsteps, nparticles, 6)),
        }

        def _step(i, data):
            self, result = data
            self.mpm_scheme.compute_nodal_kinematics()
            self.mpm_scheme.precompute_stress_strain()
            self.mpm_scheme.compute_forces(gravity, i)
            self.mpm_scheme.compute_particle_kinematics()
            self.mpm_scheme.postcompute_stress_strain()

            idu = 0
            for j in range(len(self.mesh.particles)):
                idl = 0 if j == 0 else len(self.mesh.particles[j - 1])
                idu += len(self.mesh.particles[j])
                result["position"] = (
                    result["position"]
                    .at[i, idl:idu, :]
                    .set(self.mesh.particles[j].loc.squeeze())
                )
                result["velocity"] = (
                    result["velocity"]
                    .at[i, idl:idu, :]
                    .set(self.mesh.particles[j].velocity.squeeze())
                )
                result["stress"] = (
                    result["stress"]
                    .at[i, idl:idu, :]
                    .set(self.mesh.particles[j].stress[:, :, 0].squeeze())
                )
                result["strain"] = (
                    result["strain"]
                    .at[i, idl:idu, :]
                    .set(self.mesh.particles[j].strain[:, :, 0].squeeze())
                )
            return (self, result)

        _, result = lax.fori_loop(0, nsteps, _step, (self, result))
        result = {k: jnp.asarray(v) for k, v in result.items()}
        return result
