from __future__ import annotations

from typing import TYPE_CHECKING

from jax.typing import ArrayLike

if TYPE_CHECKING:
    import jax.numpy as jnp
    from diffmpm.mesh import _MeshBase

import abc

_schemes = ("usf", "usl")

from diffmpm.node import reset_node_state


class _MPMScheme(abc.ABC):
    def __init__(self, mesh, dt, velocity_update):
        self.mesh = mesh
        self.velocity_update = velocity_update
        self.dt = dt

    def compute_nodal_kinematics(self):
        """Compute nodal kinematics - map mass and momentum to mesh nodes."""
        self.mesh.elements.nodes = reset_node_state(self.mesh.elements.nodes)
        self.mesh.apply_on_elements("set_particle_element_ids")
        self.mesh.apply_on_particles("update_natural_coords")
        self.mesh.apply_on_elements("compute_nodal_mass")
        self.mesh.apply_on_elements("compute_nodal_momentum")
        self.mesh.apply_on_elements("compute_velocity")
        self.mesh.apply_on_elements("apply_boundary_constraints")

    def compute_stress_strain(self):
        """Compute stress and strain on the particles."""
        self.mesh.apply_on_particles("compute_strain", args=(self.dt,))
        self.mesh.apply_on_particles("update_volume")
        self.mesh.apply_on_particles("compute_stress")

    def compute_forces(self, gravity: ArrayLike, step: int):
        """Compute the forces acting in the system.

        Parameters
        ----------
        gravity: ArrayLike
            Gravity present in the system. This should be an array equal
            with shape `(1, ndim)` where `ndim` is the dimension of the
            simulation.
        step: int
            Current step being simulated.
        """
        self.mesh.apply_on_elements("compute_external_force")
        self.mesh.apply_on_elements("compute_body_force", args=(gravity,))
        self.mesh.apply_traction_on_particles(step * self.dt)
        self.mesh.apply_on_elements(
            "apply_concentrated_nodal_forces", args=(step * self.dt,)
        )
        self.mesh.apply_on_elements("compute_internal_force")
        # self.mesh.apply_on_elements("apply_force_boundary_constraints")

    def compute_particle_kinematics(self):
        """Compute particle location, acceleration and velocity."""
        self.mesh.apply_on_elements("update_nodal_acceleration", args=(self.dt,))
        self.mesh.apply_on_elements("update_nodal_velocity", args=(self.dt,))
        self.mesh.apply_on_elements("update_nodal_momentum", args=(self.dt,))
        self.mesh.apply_on_particles(
            "update_position_velocity",
            args=(self.dt, self.velocity_update),
        )
        # TODO: Apply particle velocity constraints.

    @abc.abstractmethod
    def precompute_stress_strain(self):
        ...

    @abc.abstractmethod
    def postcompute_stress_strain(self):
        ...


class USF(_MPMScheme):
    """USF Scheme solver."""

    def __init__(self, mesh: _MeshBase, dt: float, velocity_update: bool):
        """Initialize USF Scheme solver.

        Parameters
        ----------
        mesh: _MeshBase
            A `diffmpm.Mesh` object that contains the elements that form
            the underlying mesh used to solve the simulation.
        dt: float
            Timestep used in the simulation.
        velocity_update: bool
            Flag to control if velocity should be updated using nodal
            velocity or interpolated nodal acceleration. If `True`, nodal
            velocity is used, else nodal acceleration. Default `False`.
        """
        super().__init__(mesh, dt, velocity_update)

    def precompute_stress_strain(self):
        """Compute stress and strain on particles."""
        self.compute_stress_strain()

    def postcompute_stress_strain(self):
        """Compute stress and strain on particles. (Empty call for USF)."""
        pass


class USL(_MPMScheme):
    """USL Scheme solver."""

    def __init__(self, mesh, dt, velocity_update):
        """Initialize USL Scheme solver.

        Parameters
        ----------
        mesh: _MeshBase
            A `diffmpm.Mesh` object that contains the elements that form
            the underlying mesh used to solve the simulation.
        dt: float
            Timestep used in the simulation.
        velocity_update: bool
            Flag to control if velocity should be updated using nodal
            velocity or interpolated nodal acceleration. If `True`, nodal
            velocity is used, else nodal acceleration. Default `False`.
        """
        super().__init__(mesh, dt, velocity_update)

    def precompute_stress_strain(self):
        """Compute stress and strain on particles. (Empty call for USL)."""
        pass

    def postcompute_stress_strain(self):
        """Compute stress and strain on particles."""
        self.compute_stress_strain()
