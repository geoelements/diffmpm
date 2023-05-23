import jax.numpy as jnp
from jax import lax
from diffmpm.scheme import _schemes, USF, USL
from tqdm import tqdm


class MPMExplicit:
    def __init__(self, mesh, dt, scheme="usf"):
        if scheme == "usf":
            self.mpm_scheme = USF(mesh, dt)
        elif scheme == "usl":
            self.mpm_scheme = USL(mesh, dt)
        else:
            raise ValueError(
                f"Please select scheme from {_schemes}. Found {scheme}"
            )
        self.mesh = mesh
        self.dt = dt

    def solve(self, nsteps: int, gravity: float):
        # self.initialize_materials()
        # self.initialize_mesh()
        # self.initialize_particles()
        result = {"position": [], "velocity": []}
        for step in tqdm(range(nsteps)):
            self.mpm_scheme.compute_nodal_kinematics()
            self.mpm_scheme.precompute_stress_strain()
            self.mpm_scheme.compute_forces(gravity)
            self.mpm_scheme.compute_particle_kinematics()
            self.mpm_scheme.postcompute_stress_strain()
            for pset in self.mesh.particles:
                result["position"].append(pset.loc)
                result["velocity"].append(pset.velocity)

        result = {k: jnp.asarray(v) for k, v in result.items()}
        return result

    def solve_jit(self, nsteps: int, gravity: float):
        # self.initialize_materials()
        # self.initialize_mesh()
        # self.initialize_particles()
        nparticles = sum(pset.loc.shape[0] for pset in self.mesh.particles)
        result = {
            "position": jnp.zeros((nsteps, nparticles)),
            "velocity": jnp.zeros((nsteps, nparticles)),
        }

        def _step(i, data):
            self, result = data
            self.mpm_scheme.compute_nodal_kinematics()
            self.mpm_scheme.precompute_stress_strain()
            self.mpm_scheme.compute_forces(gravity)
            self.mpm_scheme.compute_particle_kinematics()
            self.mpm_scheme.postcompute_stress_strain()

            def _add_res(j, data):
                particles, result = data
                result["position"] = (
                    result["position"].at[i, :].set(particles[j].loc.squeeze())
                )
                result["velocity"] = (
                    result["velocity"]
                    .at[i, :]
                    .set(particles[j].velocity.squeeze())
                )
                return (particles, result)

            _, result = lax.fori_loop(
                0,
                len(self.mesh.particles),
                _add_res,
                (self.mesh.particles, result),
            )
            return (self, result)

        _, result = lax.fori_loop(0, nsteps, _step, (self, result))
        result = {k: jnp.asarray(v) for k, v in result.items()}
        return result
