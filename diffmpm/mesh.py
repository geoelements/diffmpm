import jax.numpy as jnp

from diffmpm import element
from diffmpm import node
from diffmpm import particle

class Mesh1D:
    def __init__(self, domain_size, nelements):
        
        self.nodes = []       # List of nodes
        self.elements = []    # List of elements
        self.particles = []   # List of particles
        
        self.nelements = nelements    # Number of elements in mesh
        self.ppelem = 2       # particles per element

        # Subsequent elements
        for i in range(1, nelements):
            # Create element
            el = element.Bar1D()
            el.id = i
            
            # Node 0 
            if i == 0: # First element
                el.node1 = node.Node1D()
                el.node1.id = 0
            else: # Subsequent elements connected to previous node
                el.node1=self.elements[i-1].node2
            self.nodes.append(el.node1)

            # Node 1
            el.node2 = node.Node1D()
            el.node2.id = len(self.nodes)
            self.nodes.append(el.node2)

            # Element length and coordinates
            length = domain_size/nelements
            el.size = jnp.array([length])
            el.node1.x = jnp.array([i * length])
            el.node2.x = jnp.array([el.node1.x + length])
            
            self.elements.append(el)
            