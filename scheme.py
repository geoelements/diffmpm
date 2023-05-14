import abc

_schemes = ("usf", "usl")


class _MPMScheme(abc.ABC):
    def __init__(self, mesh, dt):
        self.mesh = mesh
        self.dt = dt

    def compute_nodal_kinematics(self):
        # self.mesh.elements.update_nodal_mass(self.mesh.particles)
        self.mesh.apply("node_update_mass")
        # self.mesh.elements.update_nodal_momentum(self.mesh.particles)
        self.mesh.apply("node_update_momentum")
        # TODO: Apply boundary conditions.

    def compute_stress_strain(self):
        # self.mesh.particles.compute_strain(self.dt)
        self.mesh.apply("particle_compute_strain")
        # self.mesh.particles.compute_stress()
        self.mesh.apply("particle_compute_stress")

    def compute_forces(self, gravity):
        # self.mesh.elements.compute_body_force(self.mesh.particles, gravity)
        self.mesh.apply("node_update_external_force")
        # self.mesh.elements.compute_internal_force(self.mesh.particles)
        self.mesh.apply("node_update_internal_force")

    def compute_particle_kinematics(self):
        # self.mesh.particles.update_position_velocity(
        #     self.mesh.elements, self.dt
        # )
        self.mesh.apply("particle_update_position_velocity")

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
