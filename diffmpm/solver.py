from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Callable, Optional

import jax.numpy as jnp
from jax import lax, profiler
from jax.experimental.host_callback import id_tap
from jax.tree_util import register_pytree_node_class
from jax.typing import ArrayLike

from diffmpm.pbar import loop_tqdm
from diffmpm.scheme import USF, USL, _MPMScheme, _schemes

if TYPE_CHECKING:
    from diffmpm.mesh import _MeshBase


@register_pytree_node_class
class MPMExplicit:
    """A class to implement the fully explicit MPM."""

    __particle_props = ("loc", "velocity", "stress", "strain")

    def __init__(
        self,
        mesh: _MeshBase,
        dt: float,
        scheme: str = "usf",
        velocity_update: bool = False,
        sim_steps: int = 1,
        out_steps: int = 1,
        out_dir: str = "results/",
        writer_func: Optional[Callable] = None,
    ) -> None:
        """Create an `MPMExplicit` object.

        This can be used to solve a given configuration of an MPM
        problem.

        Parameters
        ----------
        mesh: _MeshBase
            A `diffmpm.Mesh` object that contains the elements that form
            the underlying mesh used to solve the simulation.
        dt: float
            Timestep used in the simulation.
        scheme: str
            The MPM Scheme type used for the simulation. Can be one of
            `"usl"` or `"usf"`. Default set to `"usf"`.
        velocity_update: bool
            Flag to control if velocity should be updated using nodal
            velocity or interpolated nodal acceleration. If `True`, nodal
            velocity is used, else nodal acceleration. Default `False`.
        sim_steps: int
            Number of steps to run the simulation for. Default set to 1.
        out_steps: int
            Frequency with which to store the results. For example, if
            set to 5, the result at every 5th step will be stored. Default
            set to 1.
        out_dir: str
            Path to the output directory where results are stored.
        writer_func: Callable, None
            Function that is used to write the state in the output
            directory.
        """

        if scheme == "usf":
            self.mpm_scheme: _MPMScheme = USF(mesh, dt, velocity_update)  # type: ignore
        elif scheme == "usl":
            self.mpm_scheme: _MPMScheme = USL(mesh, dt, velocity_update)  # type: ignore
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

    def _jax_writer(self, func, args):
        id_tap(func, args)

    def solve(self, gravity: ArrayLike):
        """Non-JIT solve method.

        This method runs the entire simulation for the defined number
        of steps.

        .. note::
            This is mainly used for debugging and might be removed in
            future versions or moved to the JIT solver.

        Parameters
        ----------
        gravity: ArrayLike
            Gravity present in the system. This should be an array equal
            with shape `(1, ndim)` where `ndim` is the dimension of the
            simulation.

        Returns
        -------
        dict
            A dictionary of `ArrayLike` arrays corresponding to the
            all states of the simulation after completing all steps.
        """
        from collections import defaultdict

        from tqdm import tqdm  # type: ignore

        result = defaultdict(list)
        for step in tqdm(range(self.sim_steps)):
            breakpoint()
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

        result_arr = {k: jnp.asarray(v) for k, v in result.items()}
        return result_arr

    def solve_jit(self, gravity: ArrayLike) -> dict:
        """Solver method that runs the simulation.

        This method runs the entire simulation for the defined number
        of steps.

        Parameters
        ----------
        gravity: ArrayLike
            Gravity present in the system. This should be an array equal
            with shape `(1, ndim)` where `ndim` is the dimension of the
            simulation.

        Returns
        -------
        dict
            A dictionary of `jax.numpy` arrays corresponding to the
            final state of the simulation after completing all steps.
        """

        @loop_tqdm(self.sim_steps, print_rate=1)
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
                self._jax_writer(
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

        # with profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
        #     self = lax.fori_loop(0, self.sim_steps, _step, self)
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
