import jax.numpy as jnp

class Particle1D:
    def __init__(self, mass, x, material):
        self.id = None
        
        self.mass = mass
        self.x  = x 
        self.material = material 
        self.density = material.density 

        self.velocity  = jnp.array([0])
        self.stress    = jnp.array([0])  
        self.dstrain   = jnp.array([0])
        self.momentum  = jnp.array([0])
