import jax.numpy as jnp
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
    position : array_like
        Position of all the nodes.
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

    def __init__(self, nnodes, position, velocity, mass, momentum, f_int, f_ext, f_damp):
        """
        Parameters
        ----------
        nnodes : int
            Number of nodes stored.
        position : array_like
            Position of all the nodes.
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
        self.nnodes = nnodes
        self.position = position
        self.velocity = velocity
        self.mass = mass
        self.momentum = momentum
        self.f_int = f_int
        self.f_ext = f_ext
        self.f_damp = f_damp
        return

    def tree_flatten(self):
        children = (
            self.position,
            self.velocity,
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
        return cls(*aux_data, *children)

    def reset_values(self):
        self.velocity = self.velocity.at[:].set(0)
        self.mass = self.mass.at[:].set(0)
        self.momentum = self.momentum.at[:].set(0)
        self.f_int = self.f_int.at[:].set(0)
        self.f_ext = self.f_ext.at[:].set(0)
        self.f_damp = self.f_damp.at[:].set(0)

    def __len__(self):
        return self.nnodes

    def __repr__(self):
        return f"Nodes(n={self.nnodes})"

    def get_total_force(self):
        return self.f_int + self.f_ext + self.f_damp
