import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from diffmpm.material import Material

from typing import Callable


# @register_pytree_node_class
class Particles:
    def __init__(
        self,
        loc: jnp.ndarray,
        material: Material,
        cell_ids: jnp.ndarray,
        shapefn: Callable,
    ):
        """
        Initialise a container of particles.

        Arguments
        ---------
        loc: jax.numpy.ndarray
            Location of the particles. Expected shape (nparticles, ndim)
        material: diffmpm.material.Material
            Type of material for the set of particles.
        cell_ids: jax.numpy.ndarray
            The cell ids that the particles belong to. This contains
        information that will make sense only with the information of
        the mesh that is being considered.
        shapefn: Callable
            Shape function used by the cells that the particles are in.
        """
        self.material: Material = material
        self.cell_ids: jnp.ndarray = cell_ids
        self.loc: jnp.ndarray = loc
        self.initialize()

    def initialize(self, shapefn: Callable):
        """
        Initialize the particle properties.

        Arguments
        ---------
        shapefn: Callable
            A function used by the mesh cells to map the particles to
        their reference coordinates (xi).
        """
        self.mass = jnp.zeros_like((self.loc.shape[0], 1))
        self.volume = jnp.zeros_like(self.mass)
        self.velocity = jnp.zeros_like(self.loc)
        self.acceleration = jnp.zeros_like(self.loc)
        self.stress = jnp.zeros_like(self.loc)
        self.strain = jnp.zeros_like(self.loc)
        self.strain_rate = jnp.zeros(self.loc)
        self.dstrain = jnp.zeros_like(self.loc)
        self.reference_loc = shapefn(self.loc)

    def __len__(self):
        """Set length of the class as number of particles."""
        return self.loc.shape[0]

    def __repr__(self):
        """Informative repr showing number of particles."""
        return f"Particles(nparticles={len(self)})"
