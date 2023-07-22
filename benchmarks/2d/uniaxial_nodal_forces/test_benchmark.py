import os
from pathlib import Path

import jax

jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp

from diffmpm.mpm import MPM


def test_benchmarks():
    curr_filepath = Path(__file__).absolute()
    curr_dir = curr_filepath.parent
    os.chdir(curr_dir)
    mpm = MPM("mpm-nodal-forces.toml")
    mpm.solve()

    result = jnp.load("results/uniaxial-nodal-forces/particles_0300.npz")
    ## Step 300
    assert jnp.round(result["stress"][0, :, 0].min() - 0.5925210678182377, 6) == 0.0
    assert jnp.round(result["stress"][0, :, 0].max() - 0.5974539476363379, 6) == 0.0

    ## Step 510
    result = jnp.load("results/uniaxial-nodal-forces/particles_0510.npz")
    assert jnp.round(result["stress"][0, :, 0].min() - 1.0026665338366039, 5) == 0.0
    assert jnp.round(result["stress"][0, :, 0].max() - 1.0112231542459431, 5) == 0.0

    ## Step 750
    result = jnp.load("results/uniaxial-nodal-forces/particles_0750.npz")
    assert jnp.round(result["stress"][0, :, 0].min() - 1.0000053000532143, 5) == 0.0
    assert jnp.round(result["stress"][0, :, 0].max() - 1.0000225119807862, 5) == 0.0

    ## Step 990
    result = jnp.load("results/uniaxial-nodal-forces/particles_0990.npz")
    assert jnp.round(result["stress"][0, :, 0].min() - 0.9999990078443788, 5) == 0.0
    assert jnp.round(result["stress"][0, :, 0].max() - 0.9999990292713694, 5) == 0.0
