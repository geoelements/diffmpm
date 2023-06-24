import sys
import jax.numpy as jnp
import matplotlib.pyplot as plt
from diffmpm.solver import MPM

mpm = MPM(sys.argv[1])
result = mpm.solve()
print(result["stress"][-1][:, :2])
