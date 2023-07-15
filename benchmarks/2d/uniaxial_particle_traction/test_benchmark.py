import os
from pathlib import Path

import jax.numpy as jnp

from diffmpm.mpm import MPM


def test_benchmarks():
    curr_filepath = Path(__file__).absolute()
    curr_dir = curr_filepath.parent
    os.chdir(curr_dir)
    mpm = MPM("mpm-particle-traction.toml")
    mpm.solve()
    result = jnp.load("results/uniaxial-particle-traction/particles_0300.npz")
    ## Step 300
    assert jnp.round(result["stress"][0, :, 0].min() - 0.4450086768966724, 5) == 0.0
    assert jnp.round(result["stress"][0, :, 0].max() - 0.5966527842046769, 5) == 0.0

    ## Step 510
    result = jnp.load("results/uniaxial-particle-traction/particles_0510.npz")
    assert jnp.round(result["stress"][0, :, 0].min() - 0.7528092313640623, 5) == 0.0
    assert jnp.round(result["stress"][0, :, 0].max() - 1.0109599915279937, 5) == 0.0

    ## Step 750
    result = jnp.load("results/uniaxial-particle-traction/particles_0750.npz")
    assert jnp.round(result["stress"][0, :, 0].min() - 0.7500090055681591, 5) == 0.0
    assert jnp.round(result["stress"][0, :, 0].max() - 1.0000224746314728, 5) == 0.0

    ## Step 990
    result = jnp.load("results/uniaxial-particle-traction/particles_0990.npz")
    assert jnp.round(result["stress"][0, :, 0].min() - 0.750002924022295, 5) == 0.0
    assert jnp.round(result["stress"][0, :, 0].max() - 0.9999997782938734, 5) == 0.0
