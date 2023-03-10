import jax.numpy as jnp
from jax import vmap, lax
from diffmpm.node import Nodes
from diffmpm.particle import Particles
from diffmpm.shapefn import ShapeFn


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
        temp_px = jnp.linspace(0, self.element_length, self.ppe + 1)
        if type == "uniform":
            pmass = self.element_length * self.material.density / self.ppe
            element_particle_x = (temp_px[1:] + temp_px[:-1]) / 2
            particles_x = jnp.hstack(
                [(x + element_particle_x) for x in self.nodes.position[:-1]]
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
        return self.nodes.position[jnp.asarray([element_idx, element_idx + 1])]

    def _get_element_node_vel(self, element_idx):
        """
        Given an element at index `element_idx`, return the
        mapping node coordinates for that element.
        """
        return self.nodes.velocity[element_idx, element_idx + 1]

    def _update_particle_element_ids(self):
        """
        Find the element that the particles belong to.

        If the particle doesn't lie between the boundaries of any
        element, it sets the element index to -1.
        """

        def f(x):
            idl = (
                len(self.nodes.position)
                - 1
                - jnp.asarray(self.nodes.position[::-1] <= x).nonzero(
                    size=1, fill_value=-1
                )[0][-1]
            )
            idg = (
                jnp.asarray(self.nodes.position > x).nonzero(
                    size=1, fill_value=-1
                )[0][0]
                - 1
            )
            return (idl, idg)

        ids = vmap(f)(self.particles.x)
        self.particles.element_ids = jnp.where(
            ids[0] == ids[1], ids[0], jnp.ones_like(ids[0]) * -1
        )

    def _update_particle_natural_coords(self):
        """
        Update natural coordinates for the particles.

        Whenever the particles' physical coordinates change, their
        natural coordinates need to be updated. This function updates
        the natural coordinates of the particles based on the element
        a particle is a part of. The update formula is

        :math:`\xi = \frac{(x - x_{n_0}) 2}{l} - 1`

        If a particle is not in any element (element_id = -1), its
        natural coordinate is set to 0.
        """
        t = self.nodes.position[self.particles.element_ids]
        t = jnp.where(
            self.particles.element_ids == -1,
            self.particles.x - self.element_length / 2,
            t,
        )
        xi_coords = (self.particles.x - t) * 2 / self.element_length - 1
        self.particles.xi = xi_coords

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
