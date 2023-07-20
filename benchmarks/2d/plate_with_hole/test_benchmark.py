import os
from pathlib import Path
import jax.numpy as jnp
from diffmpm import MPM


def test_benchmarks():
    curr_filepath = Path(__file__).absolute()
    curr_dir = curr_filepath.parent
    os.chdir(curr_dir)
    mpm = MPM("plate-with-hole.toml")
    mpm.solve()

    result = jnp.load("results/plate-with-a-hole/particles_299999.npz")
    true_stress_yy = -1.11
    true_stress_xx = 0.0

    assert jnp.round(result["stress"][0, :, 1].max() - true_stress_yy, 8) == 0.0
    assert jnp.round(result["stress"][0, :, 0].max() - true_stress_xx, 8) == 0.0
