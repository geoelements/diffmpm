from typing import Callable

import jax.numpy as jnp
from jax import lax, vmap
from jax.tree_util import register_pytree_node_class


# @register_pytree_node_class
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

    def __init__(self, nnodes, loc):
        """
        Initialize container for Nodes.

        Parameters
        ----------
        nnodes : int
            Number of nodes stored.
        loc : array_like
            Locations of all the nodes. Expected shape (nnodes, 1, ndim)
        """
        self.nnodes = nnodes
        if len(loc.shape) != 3:
            raise ValueError(
                f"`loc` should be of size (nnodes, ndim); found {loc.shape}"
            )
        self.loc = jnp.asarray(loc, dtype=jnp.float32)
        self.velocity = jnp.zeros_like(self.loc, dtype=jnp.float32)
        self.acceleration = jnp.zeros_like(self.loc, dtype=jnp.float32)
        self.mass = jnp.zeros((self.loc.shape[0], 1, 1), dtype=jnp.float32)
        self.momentum = jnp.zeros_like(self.loc, dtype=jnp.float32)
        self.f_int = jnp.zeros_like(self.loc, dtype=jnp.float32)
        self.f_ext = jnp.zeros_like(self.loc, dtype=jnp.float32)
        self.f_damp = jnp.zeros_like(self.loc, dtype=jnp.float32)
        return

    # def tree_flatten(self):
    #     """Helper method for registering class as Pytree type."""
    #     children = (
    #         self.position,
    #         self.velocity,
    #         self.mass,
    #         self.momentum,
    #         self.f_int,
    #         self.f_ext,
    #         self.f_damp,
    #     )
    #     aux_data = (self.nnodes,)
    #     return (children, aux_data)

    # @classmethod
    # def tree_unflatten(cls, aux_data, children):
    #     return cls(*aux_data, *children)

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

    def compute_acceleration_velocity(self, dt):
        """Calculate the nodal acceleration and velocities."""
        total_force = self.get_total_force()
        self.acceleration = self.acceleration.at[:].set(
            jnp.divide(total_force, self.mass)
        )

        self.velocity = self.velocity.at[:].add(self.acceleration * dt)
