from typing import Callable

import jax.numpy as jnp
from jax import jit, vmap
from jax.tree_util import register_pytree_node_class

from diffmpm.element import _Element
from diffmpm.material import Material


# @register_pytree_node_class
class Particles:
    def __init__(
        self,
        loc: jnp.ndarray,
        material: Material,
        element_ids: jnp.ndarray,
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
        element_ids: jax.numpy.ndarray
            The element ids that the particles belong to. This contains
        information that will make sense only with the information of
        the mesh that is being considered.
        shapefn: Callable
            Shape function used by the elements that the particles are in.
        """
        self.material: Material = material
        self.element_ids: jnp.ndarray = element_ids
        if len(loc.shape) != 2:
            raise ValueError(
                f"`loc` should be of size (nparticles, ndim); found {loc.shape}"
            )
        self.loc: jnp.ndarray = loc
        self.initialize(shapefn)

    def initialize(self, shapefn: Callable):
        """
        Initialize the particle properties.

        Arguments
        ---------
        shapefn: Callable
            A function used by the mesh elements to map the particles to
        their reference coordinates (xi).
        """
        self.mass = jnp.zeros((self.loc.shape[0], 1))
        self.volume = jnp.zeros_like(self.mass)
        self.velocity = jnp.zeros_like(self.loc)
        self.acceleration = jnp.zeros_like(self.loc)
        self.stress = jnp.zeros_like(self.loc)
        self.strain = jnp.zeros_like(self.loc)
        self.strain_rate = jnp.zeros_like(self.loc)
        self.dstrain = jnp.zeros_like(self.loc)
        self.reference_loc = jnp.zeros_like(self.loc)

    def __len__(self):
        """Set length of the class as number of particles."""
        return self.loc.shape[0]

    def __repr__(self):
        """Informative repr showing number of particles."""
        return f"Particles(nparticles={len(self)})"

    def update_particle_element_ids(self, elements: _Element):
        """
        Update the element IDs for the particles.

        If the particle doesn't lie between the boundaries of any
        element, it sets the element index to -1.
        """

        @jit
        def f(x):
            idl = (
                len(elements.nodes.loc)
                - 1
                - jnp.asarray(elements.nodes.loc[::-1] <= x).nonzero(
                    size=1, fill_value=-1
                )[0][-1]
            )
            idg = (
                jnp.asarray(elements.nodes.loc > x).nonzero(
                    size=1, fill_value=-1
                )[0][0]
                - 1
            )
            return (idl, idg)

        ids = vmap(f)(self.loc)
        self.element_ids = jnp.where(
            ids[0] == ids[1], ids[0], jnp.ones_like(ids[0]) * -1
        )
