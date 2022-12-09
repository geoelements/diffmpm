import jax.numpy as jnp

class Node1D:
    """1D MPM node
    Attributes:
        id: Index of node class.
        x: Location of node.
        mass: Mass at node.
        velocity: Velocity at node.
        momentum: Momentum at node.
        f_int: Internal force.
        f_ext: External force.
    """

    def __init__(self):
        self.id = None
        self.x = jnp.array([0])
        self.mass = jnp.array([0])
        self.velocity = jnp.array([0])
        self.momentum = jnp.array([0])
        self.f_int = jnp.array([0])
        self.f_ext = jnp.array([0])

