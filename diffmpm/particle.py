import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class Particles:
    """
    Container class for particles on a mesh.
    """

    def __init__(
        self,
        mass,
        x,
        xi,
        density,
        element_ids,
        velocity,
        volume,
        stress,
        strain,
        dstrain,
        f_ext,
        ppe=1,
        nelements=1,
        nparticles=1,
        material=None,
        ptype="uniform",
    ):
        """
        Construct a container for particles.

        Arguments
        ---------
        mass : float, array_like
            Mass of each particle. Can be a float or an array for mass
        of each particle.
        x : array_like
            Position of particles in physical coordinates.
        xi : array_like
            Position of particles in natural coordinates.
        material : diffmpm.material.Material
            Material type of the mesh the particles are a part of.
        density : float, array_like
            Density of each particle. Can be a float or an array for
        density of each particle.
        ppe : int
            Number of particles per element.
        nelements : int
            Number of elements that contain the particles.
        element_ids : array_like
            Ids of the elements that each particle is a part of.
        """
        self.material = material
        self.ppe = ppe
        self.nparticles = ppe * nelements
        self.x = x
        self.xi = xi
        self.element_ids = element_ids
        self.mass = (
            mass if not jnp.isscalar(mass) else jnp.ones(self.nparticles) * mass
        )
        self.density = (
            density
            if not jnp.isscalar(density)
            else jnp.ones(self.nparticles) * density
        )

        self.velocity = velocity
        self.volume = volume
        self.stress = stress
        self.strain = strain
        self.dstrain = dstrain
        self.f_ext = f_ext

        return

    def __len__(self):
        return self.nparticles

    def __repr__(self):
        return f"Particles(nparticles={self.nparticles})"

    def tree_flatten(self):
        children = (
            self.mass,
            self.x,
            self.xi,
            self.density,
            self.element_ids,
            self.velocity,
            self.volume,
            self.stress,
            self.strain,
            self.dstrain,
            self.f_ext,
        )
        aux_data = {
            "material": self.material,
            "ppe": self.ppe,
            "nelements": self.nparticles // self.ppe,
            "nparticles": self.nparticles,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            *children[:5],
            *children[5:],
            **aux_data,
        )
