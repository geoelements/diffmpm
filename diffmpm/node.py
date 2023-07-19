from typing import Optional, Sized, Tuple

import jax.numpy as jnp
from jax import jit
from jax.tree_util import register_pytree_node_class
from jax.typing import ArrayLike


@register_pytree_node_class
class Nodes(Sized):
    """Nodes container class.

    Keeps track of all values required for nodal points.

    Attributes
    ----------
    nnodes : int
        Number of nodes stored.
    loc : ArrayLike
        Location of all the nodes.
    velocity : array_like
        Velocity of all the nodes.
    mass : ArrayLike
        Mass of all the nodes.
    momentum : array_like
        Momentum of all the nodes.
    f_int : ArrayLike
        Internal forces on all the nodes.
    f_ext : ArrayLike
        External forces present on all the nodes.
    f_damp : ArrayLike
        Damping forces on the nodes.
    """

    def __init__(
        self,
        nnodes: int,
        loc: ArrayLike,
        initialized: Optional[bool] = None,
        data: Tuple[ArrayLike, ...] = tuple(),
    ):
        """Initialize container for Nodes.

        Parameters
        ----------
        nnodes : int
            Number of nodes stored.
        loc : ArrayLike
            Locations of all the nodes. Expected shape (nnodes, 1, ndim)
        initialized: bool
            `False` if node property arrays like mass need to be initialized.
            If `True`, they are set to values from `data`.
        data: tuple
            Tuple of length 7 that sets arrays for mass, density, volume,
            and forces. Mainly used by JAX while unflattening.
        """
        self.nnodes = nnodes
        loc = jnp.asarray(loc, dtype=jnp.float32)
        if loc.ndim != 3:
            raise ValueError(
                f"`loc` should be of size (nnodes, 1, ndim); found {loc.shape}"
            )
        self.loc = loc

        if initialized is None:
            self.velocity = jnp.zeros_like(self.loc, dtype=jnp.float32)
            self.acceleration = jnp.zeros_like(self.loc, dtype=jnp.float32)
            self.mass = jnp.ones((self.loc.shape[0], 1, 1), dtype=jnp.float32)
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
            ) = data  # type: ignore
        self.initialized = True

    def tree_flatten(self):
        """Flatten class as Pytree type."""
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
        """Unflatten class from Pytree type."""
        return cls(aux_data[0], children[0], initialized=children[1], data=children[2:])

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

    @jit
    def get_total_force(self):
        """Calculate total force on the nodes."""
        return self.f_int + self.f_ext + self.f_damp
