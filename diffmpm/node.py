import jax.numpy as jnp


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
    def __init__(self, n):
        """
        Parameters
        ----------
        n : int
            Number of nodes in the mesh.
        """
        self.nnodes = n
        self.position = jnp.zeros(n)
        self.velocity = jnp.zeros(n)
        self.mass = jnp.zeros(n)
        self.momentum = jnp.zeros(n)
        self.f_int = jnp.zeros(n)
        self.f_ext = jnp.zeros(n)
        self.f_damp = jnp.zeros(n)
        return


    def _reset_values(self):
        self.velocity.fill(0)
        self.mass.fill(0)
        self.momentum.fill(0)
        self.f_int.fill(0)
        self.f_ext.fill(0)
        self.f_damp.fill(0)
