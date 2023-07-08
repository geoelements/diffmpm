import abc
from typing import Iterable

import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from diffmpm.element import _Element
from diffmpm.particle import Particles


class _MeshBase(abc.ABC):
    """
    Base class for Meshes.

    Note: If attributes other than elements and particles are added
    then the child class should also implement `tree_flatten` and
    `tree_unflatten` correctly or that information will get lost.
    """

    def __init__(self, config: dict):
        """Initialize mesh using configuration."""
        self.particles: Iterable[Particles, ...] = config["particles"]
        self.elements: _Element = config["elements"]
        self.particle_tractions = config["particle_surface_traction"]

    @property
    @abc.abstractmethod
    def ndim(self):
        ...

    # TODO: Convert to using jax directives for loop
    def apply_on_elements(self, function, args=()):
        f = getattr(self.elements, function)
        for particle_set in self.particles:
            f(particle_set, *args)

    # TODO: Convert to using jax directives for loop
    def apply_on_particles(self, function, args=()):
        for particle_set in self.particles:
            f = getattr(particle_set, function)
            f(self.elements, *args)

    def apply_traction_on_particles(self, curr_time):
        self.apply_on_particles("zero_traction")
        for ptraction in self.particle_tractions:
            factor = ptraction.function.value(curr_time)
            traction_val = factor * ptraction.traction
            for i, pset_id in enumerate(ptraction.pset):
                self.particles[pset_id].assign_traction(
                    ptraction.pids, ptraction.dir, traction_val
                )

        # breakpoint()
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
        """
        Initialize a 1D Mesh.

        Arguments
        ---------
        config: dict
            Configuration to be used for initialization. It _should_
        contain `elements` and `particles` keys.
        """
        super().__init__(config)

    @property
    def ndim(self):
        return 1


@register_pytree_node_class
class Mesh2D(_MeshBase):
    """1D Mesh class with nodes, elements, and particles."""

    def __init__(self, config: dict):
        """
        Initialize a 2D Mesh.

        Arguments
        ---------
        config: dict
            Configuration to be used for initialization. It _should_
        contain `elements` and `particles` keys.
        """
        super().__init__(config)

    @property
    def ndim(self):
        return 2


if __name__ == "__main__":
    from diffmpm.element import Linear1D
    from diffmpm.material import SimpleMaterial
    from diffmpm.utils import _show_example

    particles = Particles(
        jnp.array([[[1]]]),
        SimpleMaterial({"E": 2, "density": 1}),
        jnp.array([0]),
    )
    elements = Linear1D(2, 1, jnp.array([0]))
    _show_example(Mesh1D({"particles": [particles], "elements": elements}))
