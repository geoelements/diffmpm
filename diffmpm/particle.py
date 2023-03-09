import jax.numpy as jnp


class Particles:
    """
    Container class for particles on a mesh.
    """

    def __init__(
        self,
        mass,
        x,
        xi,
        material,
        ppe,
        nelements,
        element_ids,
        domain_size,
        ptype="uniform",
    ):
        """
        Construct a container for particles.

        Arguments
        ---------
        mass : float, array_like
            Mass of each particle. Can be a float or an array for mass of each particle.
        x : array_like
            Position of particles in physical coordinates.
        xi : array_like
            Position of particles in natural coordinates.
        material : diffmpm.material.Material
            Material type of the mesh the particles are a part of.
        ppe : int
            Number of particles per element.
        nelements : int
            Number of elements that contain the particles.
        element_ids : array_like
            Ids of the elements that each particle is a part of.
        domain_size : float
            Size of the domain in consideration.
        """
        # self.mass = (
        #     jnp.empty(ppe * nelements).fill(mass)
        #     if jnp.isscalar(mass)
        #     else jnp.asarray(mass)
        # )
        self.material = material
        self.nparticles = ppe * nelements
        self.x = x
        self.xi = xi
        self.element_ids = element_ids
        self.mass = mass
        assert len(x) == self.nparticles

        self.velocity = jnp.zeros(self.nparticles)
        self.volume = jnp.zeros(self.nparticles)
        self.stress = jnp.zeros(self.nparticles)
        self.strain = jnp.zeros(self.nparticles)
        self.dstrain = jnp.zeros(self.nparticles)
        self.f_ext = jnp.zeros(self.nparticles)

        return
