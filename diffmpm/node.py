from typing import Callable, Tuple

import jax.numpy as jnp
from jax import lax, vmap
from jax import debug
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class Nodes:
    """
    Nodes container class.

    Keeps track of all values required for nodal points.

    Attributes
    ----------
    nnodes : int
        Number of nodes stored.
    loc : array_like
        Location of all the nodes.
    velocity : array_like
        Velocity of all the nodes.
    mass : array_like
        Mass of all the nodes.
    momentum : array_like
        Momentum of all the nodes.
    f_int : array_like
        Internal forces on all the nodes.
    f_ext : array_like
        External forces present on all the nodes.
    f_damp : array_like
        Damping forces on the nodes.
    """

    def __init__(
        self,
        nnodes: int,
        loc: jnp.ndarray,
        initialized: bool = None,
        data: Tuple[jnp.ndarray, ...] = tuple(),
    ):
        """
        Initialize container for Nodes.

        Parameters
        ----------
        nnodes : int
            Number of nodes stored.
        loc : array_like
            Locations of all the nodes. Expected shape (nnodes, 1, ndim)
        initialized: bool
            False if node property arrays like mass need to be initialized.
        If True, they are set to values from `data`.
        data: tuple
            Tuple of length 7 that sets arrays for mass, density, volume,
        """
        self.nnodes = nnodes
        if len(loc.shape) != 3:
            raise ValueError(
                f"`loc` should be of size (nnodes, 1, ndim); found {loc.shape}"
            )
        self.loc = jnp.asarray(loc, dtype=jnp.float32)

        # debug.breakpoint()
        # breakpoint()
        # _data = lax.cond(initialized, self.__set_vals__, self.initialize, data)
        # _data = lax.switch(
        #     int(initialized),
        #     (self.initialize, self.__set_vals__),
        #     data,
        # )
        if initialized is None:
            self.velocity = jnp.zeros_like(self.loc, dtype=jnp.float32)
            self.acceleration = jnp.zeros_like(self.loc, dtype=jnp.float32)
            self.mass = jnp.zeros((self.loc.shape[0], 1, 1), dtype=jnp.float32)
            self.momentum = jnp.zeros_like(self.loc, dtype=jnp.float32)
            self.f_int = jnp.zeros_like(self.loc, dtype=jnp.float32)
            self.f_ext = jnp.zeros_like(self.loc, dtype=jnp.float32)
            self.f_damp = jnp.zeros_like(self.loc, dtype=jnp.float32)
        else:
            (
                self.velocity,
                self.acceleration,
                self.mass,
                self.momentum,
                self.f_int,
                self.f_ext,
                self.f_damp,
            ) = data
        self.initialized = True

    def __set_vals__(self, data):
        # debug.breakpoint()
        # breakpoint()
        # if len(data) != 7:
        #     raise ValueError(
        #         f"Node `data` is expected to be a tuple of length 7, "
        #         f"found {len(data)}."
        #     )
        return (*data, True)

    def initialize(self, *args):
        velocity = jnp.zeros_like(self.loc, dtype=jnp.float32)
        acceleration = jnp.zeros_like(self.loc, dtype=jnp.float32)
        mass = jnp.zeros((self.loc.shape[0], 1, 1), dtype=jnp.float32)
        momentum = jnp.zeros_like(self.loc, dtype=jnp.float32)
        f_int = jnp.zeros_like(self.loc, dtype=jnp.float32)
        f_ext = jnp.zeros_like(self.loc, dtype=jnp.float32)
        f_damp = jnp.zeros_like(self.loc, dtype=jnp.float32)
        initialized = True
        return (
            velocity,
            acceleration,
            mass,
            momentum,
            f_int,
            f_ext,
            f_damp,
            initialized,
        )

    def tree_flatten(self):
        """Helper method for registering class as Pytree type."""
        children = (
            self.loc,
            self.initialized,
            self.velocity,
            self.acceleration,
            self.mass,
            self.momentum,
            self.f_int,
            self.f_ext,
            self.f_damp,
        )
        aux_data = (self.nnodes,)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            aux_data[0], children[0], initialized=children[1], data=children[2:]
        )

    def reset_values(self):
        """Reset nodal parameter values except location."""
        self.velocity = self.velocity.at[:].set(0)
        self.acceleration = self.velocity.at[:].set(0)
        self.mass = self.mass.at[:].set(0)
        self.momentum = self.momentum.at[:].set(0)
        self.f_int = self.f_int.at[:].set(0)
        self.f_ext = self.f_ext.at[:].set(0)
        self.f_damp = self.f_damp.at[:].set(0)

    def __len__(self):
        """Set length of class as number of nodes."""
        return self.nnodes

    def __repr__(self):
        """Repr containing number of nodes."""
        return f"Nodes(n={self.nnodes})"

    def get_total_force(self):
        """Calculate total force on the nodes."""
        return self.f_int + self.f_ext + self.f_damp


if __name__ == "__main__":
    from diffmpm.utils import _show_example

    _show_example(Nodes(2, jnp.array([1, 2]).reshape(2, 1, 1)))
