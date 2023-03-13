import jax.numpy as jnp
from diffmpm.shapefn import ShapeFn


def test_shapefn_1d_scalar():
    x = 0.5
    sf = ShapeFn(dim=1)
    # True values are calculated as
    # [(1-x)/2, (1+x)/2]
    true_sf = jnp.array([0.25, 0.75])
    assert jnp.alltrue(sf.shapefn(x) == true_sf)


def test_shapefn_1d_scalar_grad():
    x = 0.5
    sf = ShapeFn(dim=1)
    true_grad = jnp.array([-0.5, 0.5])
    assert jnp.all(sf.shapefn_grad(x, jnp.array([-1, 1])) == true_grad)


def test_shapefn_1d_vec():
    x = jnp.array([0.5, 0.5])
    sf = ShapeFn(dim=1)
    true_sf = jnp.array([[0.25, 0.75], [0.25, 0.75]])
    calc_sf = sf.shapefn(x)
    assert jnp.allclose(calc_sf, true_sf)


def test_shapefn_1d_vec_grad():
    x = jnp.array([0.5, 0.5])
    sf = ShapeFn(dim=1)
    true_grad = jnp.array([[-0.5, 0.5], [-0.5, 0.5]])
    calc_grad = sf.shapefn_grad(x, jnp.array([-1, 1]))
    assert jnp.allclose(calc_grad, true_grad)