import jax.numpy as jnp
from jax import vmap
from node import Nodes
from particle import Particles


class Mesh1D:
    """
    1D Mesh class with nodes, elements, and particles.
    """

    def __init__(
        self, nelements, material, domain_size, ppe=1, particle_type="uniform"
    ):
        """
        Construct a 1D Mesh.

        Arguments
        ---------
        nelements : int
            Number of elements in the mesh.
        material : diffmpm.material.Material
            Material to meshed.
        domain_size : float
            The size of the domain in consideration.
        ppe : int
            Number of particles per element in Mesh.
        """
        self.dim = 1
        self.material = material
        self.shapefn = ShapeFn(self.dim)
        self.domain_size = domain_size
        self.nelements = nelements
        self.element_length = domain_size / nelements
        self.elements = jnp.arange(nelements)
        self.nodes = Nodes(nelements + 1)
        self.nodes.position = jnp.arange(nelements + 1) * self.element_length
        self.ppe = ppe
        self.particles = self._init_particles(particle_type)
        return

    def _init_particles(self, type="uniform"):
        temp_px = jnp.arange(0, self.element_length, 1 / self.ppe)
        if type == "uniform":
            pmass = self.element_length * self.material.density / self.ppe
            element_particle_x = (temp_px[1:] + temp_px[:-1]) / 2
            particles_x = jnp.asarray(
                [(x + element_particle_x) for x in self.nodes.positions[:-1]]
            )
            particles_xi = jnp.tile(element_particle_x, self.nelements)
            particle_element_ids = jnp.repeat(
                jnp.arange(self.nelements), self.ppe
            )
            particles = Particles(
                pmass,
                particles_x,
                particles_xi,
                self.material,
                self.ppe,
                self.nelements,
                particle_element_ids,
                self.domain_size,
            )
            return particles
        else:
            raise ValueError(
                f"{type} type particle initialization not "
                f"yet supported. Please use 'uniform'."
            )

    def _get_element_node_pos(self, element_idx):
        """
        Given an element at index `element_idx`, return the
        mapping node coordinates for that element.
        """
        return self.nodes.position[element_idx, element_idx + 1]

    def _get_element_node_vel(self, element_idx):
        """
        Given an element at index `element_idx`, return the
        mapping node coordinates for that element.
        """
        return self.nodes.velocity[element_idx, element_idx + 1]

    def _update_particle_strain(self):
        """
        Calculate the strain values for particles.

        This calculation is done by mapping the nodal velocities
        with the gradient of the interpolation shape function.
        """

        nodal_coords = vmap(self._get_element_node_pos)(
            self.particles.element_ids
        )
        # particles_dndx will be of shape (nparticles, element.nnodes)
        particles_dndx = vmap(self.shapefn.shapefn_grad)(
            self.particles.xi, nodal_coords
        )
        nodal_vel = vmap(self._get_element_node_vel)(self.particles.element_ids)

        # strain rate is the row-wise sum of the matrix particles_dndx x nodal_vel
        strain_rate = jnp.sum(particles_dndx * nodal_vel, axis=0)

        self.particles.dstrain = strain_rate * dt
        self.particles.strain += self.particles.dstrain
        return

    def _update_particle_stress(self):
        self.particle.stress += self.particle.dstrain * self.material.E
        return
