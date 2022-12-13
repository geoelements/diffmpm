import jax.numpy as jnp

class Bar1D:
    """
    1D bar element with 2 nodes.

    Attributes:

    id: int
        Element index.
    
    nid1: int
        Node index 1.
    
    nid2: int
        Node index 2.

    length: float
        Element length.
    
    particles: list
        List of particles in bar.
    """

    def __init__(self):
        self.id = None
        self.node1 = None
        self.node2 = None
        self.size = jnp.array([0])
        self.particles = []