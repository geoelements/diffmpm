import jax.numpy as jnp

class LinearElastic:
    def  __init__(self, E, density):
        self.E = E
        self.density = density

    def update_stress(self, particle, dt):
        particle.stress+=particle.dstrain*self.E