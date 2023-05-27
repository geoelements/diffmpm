import abc
from diffmpm.element import _Element
from diffmpm.particle import Particles
from jax.tree_util import register_pytree_node_class

_schemes = ("usf", "usl")


class _MPMScheme(abc.ABC):
    def __init__(self, mesh, dt):
        self.mesh = mesh
        self.dt = dt

    def compute_nodal_kinematics(self):
        self.mesh.apply_on_elements("update_particle_natural_coords")
        self.mesh.apply_on_particles("set_particle_element_ids")
        self.mesh.apply_on_elements("compute_nodal_mass")
        self.mesh.apply_on_elements("compute_nodal_momentum")
        # TODO: Apply boundary conditions.
        self.mesh.apply_on_elements("apply_boundary_constraints")

    def compute_stress_strain(self):
        self.mesh.apply_on_particles("compute_strain", args=(self.dt,))
        self.mesh.apply_on_particles("compute_stress")

    def compute_forces(self, gravity):
        self.mesh.apply_on_elements("compute_body_force", args=(gravity,))
        self.mesh.apply_on_elements("compute_external_force")
        self.mesh.apply_on_elements("compute_internal_force")
        self.mesh.apply_on_elements("apply_force_boundary_constraints")

    def compute_particle_kinematics(self):
        self.mesh.apply_on_elements(
            "compute_acceleration_velocity", args=(self.dt,)
        )
        self.mesh.apply_on_particles(
            "update_position_velocity", args=(self.dt,)
        )

    @abc.abstractmethod
    def precompute_stress_strain():
        ...

    @abc.abstractmethod
    def postcompute_stress_strain():
        ...


class USF(_MPMScheme):
    """USF Scheme solver."""

    def __init__(self, mesh, dt):
        super().__init__(mesh, dt)

    def precompute_stress_strain(self):
        self.compute_stress_strain()

    def postcompute_stress_strain(self):
        pass


class USL(_MPMScheme):
    """USL Scheme solver."""

    def __init__(self, mesh, dt):
        super().__init__(mesh, dt)

    def precompute_stress_strain(self):
        pass

    def postcompute_stress_strain(self):
        self.compute_stress_strain()
