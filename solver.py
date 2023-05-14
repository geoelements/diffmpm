from diffmpm.scheme import _schemes, USF, USL


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

    def compute_stress_strain(self):
        # self.mesh.particles.compute_strain(self.mesh.elements, self.dt)
        self.mesh.apply("particle_compute_strain")
        # self.mesh.particles.update_volume()
        self.mesh.apply("particle_update_volume")
        # self.mesh.materials.compute_stress()
        self.mesh.apply("particle_compute_stress")

    def solve(self, nsteps: int, gravity: float):
        self.initialize_materials()
        self.initialize_mesh()
        self.initialize_particles()

        for step in range(nsteps):
            self.mpm_scheme.compute_nodal_kinematics()
            self.mpm_scheme.precompute_stress_strain()
            self.mpm_scheme.compute_forces(gravity)
            self.mpm_scheme.compute_particle_kinematics()
            self.mpm_scheme.postcompute_stress_strain()
