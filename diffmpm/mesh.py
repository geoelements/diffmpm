import abc
from functools import partial
from typing import Callable, Sequence, Tuple

import jax.numpy as jnp
from jax import lax
from jax.tree_util import register_pytree_node_class, tree_map

from diffmpm.element import _Element
from diffmpm.particle import Particles

__all__ = ["_MeshBase", "Mesh1D", "Mesh2D"]


class _MeshBase(abc.ABC):
    """Base class for Meshes.

    .. note::
        If attributes other than elements and particles are added
        then the child class should also implement `tree_flatten` and
        `tree_unflatten` correctly or that information will get lost.
    """

    ndim: int

    def __init__(self, config: dict):
        """Initialize mesh using configuration."""
        self.particles: Sequence[Particles] = config["particles"]
        self.elements: _Element = config["elements"]
        self.particle_tractions = config["particle_surface_traction"]

    # TODO: Convert to using jax directives for loop
    def apply_on_elements(self, function: str, args: Tuple = ()):
        """Apply a given function to elements.

        Parameters
        ----------
        function: str
            A string corresponding to a function name in `_Element`.
        args: tuple
            Parameters to be passed to the function.
        """
        f = getattr(self.elements, function)

        def _func(particles, *, func, fargs):
            func(particles, *fargs)

        partial_func = partial(_func, func=f, fargs=args)
        tree_map(
            partial_func, self.particles, is_leaf=lambda x: isinstance(x, Particles)
        )

    # TODO: Convert to using jax directives for loop
    def apply_on_particles(self, function: str, args: Tuple = ()):
        """Apply a given function to particles.

        Parameters
        ----------
        function: str
            A string corresponding to a function name in `Particles`.
        args: tuple
            Parameters to be passed to the function.
        """

        def _func(particles, *, elements, fname, fargs):
            f = getattr(particles, fname)
            f(elements, *fargs)

        partial_func = partial(
            _func, elements=self.elements, fname=function, fargs=args
        )
        tree_map(
            partial_func, self.particles, is_leaf=lambda x: isinstance(x, Particles)
        )

    def apply_traction_on_particles(self, curr_time: float):
        """Apply tractions on particles.

        Parameters
        ----------
        curr_time: float
            Current time in the simulation.
        """
        self.apply_on_particles("zero_traction")
        for ptraction in self.particle_tractions:
            factor = ptraction.function.value(curr_time)
            traction_val = factor * ptraction.traction
            for i, pset_id in enumerate(ptraction.pset):
                self.particles[pset_id].assign_traction(
                    ptraction.pids, ptraction.dir, traction_val
                )

        self.apply_on_elements("apply_particle_traction_forces")

    def tree_flatten(self):
        children = (self.particles, self.elements)
        aux_data = self.particle_tractions
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            {
                "particles": children[0],
                "elements": children[1],
                "particle_surface_traction": aux_data,
            }
        )


@register_pytree_node_class
class Mesh1D(_MeshBase):
    """1D Mesh class with nodes, elements, and particles."""

    def __init__(self, config: dict):
        """Initialize a 1D Mesh.

        Parameters
        ----------
        config: dict
            Configuration to be used for initialization. It _should_
            contain `elements` and `particles` keys.
        """
        self.ndim = 1
        super().__init__(config)


@register_pytree_node_class
class Mesh2D(_MeshBase):
    """1D Mesh class with nodes, elements, and particles."""

    def __init__(self, config: dict):
        """Initialize a 2D Mesh.

        Parameters
        ----------
        config: dict
            Configuration to be used for initialization. It _should_
            contain `elements` and `particles` keys.
        """
        self.ndim = 2
        super().__init__(config)
