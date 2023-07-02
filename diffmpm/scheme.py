import abc

_schemes = ("usf", "usl")


class _MPMScheme(abc.ABC):
    def __init__(self, mesh, dt, velocity_update):
        self.mesh = mesh
        self.velocity_update = velocity_update
        self.dt = dt

    def compute_nodal_kinematics(self):
        self.mesh.apply_on_elements("set_particle_element_ids")
        self.mesh.apply_on_particles("update_natural_coords")
        self.mesh.apply_on_elements("compute_nodal_mass")
        self.mesh.apply_on_elements("compute_nodal_momentum")
        self.mesh.apply_on_elements("compute_velocity")
        self.mesh.apply_on_elements("apply_boundary_constraints")

    def compute_stress_strain(self):
        self.mesh.apply_on_particles("compute_strain", args=(self.dt,))
        self.mesh.apply_on_particles("update_volume")
        self.mesh.apply_on_particles("compute_stress")

    def compute_forces(self, gravity, step):
        self.mesh.apply_on_elements("compute_external_force")
        self.mesh.apply_on_elements("compute_body_force", args=(gravity,))
        self.mesh.apply_traction_on_particles(step * self.dt)
        self.mesh.apply_on_elements(
            "apply_concentrated_nodal_forces", args=(step * self.dt,)
        )
        self.mesh.apply_on_elements("compute_internal_force")
        # self.mesh.apply_on_elements("apply_force_boundary_constraints")

    def compute_particle_kinematics(self):
        self.mesh.apply_on_elements(
            "update_nodal_acceleration_velocity", args=(self.dt,)
        )
        self.mesh.apply_on_particles(
            "update_position_velocity",
            args=(self.dt, self.velocity_update),
        )
        # TODO: Apply particle velocity constraints.

    @abc.abstractmethod
    def precompute_stress_strain():
        ...

    @abc.abstractmethod
    def postcompute_stress_strain():
        ...


class USF(_MPMScheme):
    """USF Scheme solver."""

    def __init__(self, mesh, dt, velocity_update):
        super().__init__(mesh, dt, velocity_update)

    def precompute_stress_strain(self):
        self.compute_stress_strain()

    def postcompute_stress_strain(self):
        pass


class USL(_MPMScheme):
    """USL Scheme solver."""

    def __init__(self, mesh, dt, velocity_update):
        super().__init__(mesh, dt, velocity_update)

    def precompute_stress_strain(self):
        pass

    def postcompute_stress_strain(self):
        self.compute_stress_strain()
