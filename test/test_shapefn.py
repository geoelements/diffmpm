import jax.numpy as jnp
from diffmpm.shapefn import Linear1DShapeFn, Linear4NodeQuad


def test_shapefn_1d_scalar():
    x = 0.5
    sf = Linear1DShapeFn(dim=1)
    # True values are calculated as
    # [(1-x)/2, (1+x)/2]
    true_sf = jnp.array([0.25, 0.75])
    assert jnp.alltrue(sf.shapefn(x) == true_sf)


def test_shapefn_1d_scalar_grad():
    x = 0.5
    sf = Linear1DShapeFn(dim=1)
    true_grad = jnp.array([-0.5, 0.5])
    assert jnp.all(sf.shapefn_grad(x, jnp.array([-1, 1])) == true_grad)


def test_shapefn_1d_vec():
    x = jnp.array([0.5, 0.5])
    sf = Linear1DShapeFn(dim=1)
    true_sf = jnp.array([[0.25, 0.75], [0.25, 0.75]])
    calc_sf = sf.shapefn(x)
    assert jnp.allclose(calc_sf, true_sf)


def test_shapefn_1d_vec_grad():
    x = jnp.array([0.5, 0.5])
    sf = Linear1DShapeFn(dim=1)
    true_grad = jnp.array([[-0.5, 0.5], [-0.5, 0.5]])
    calc_grad = sf.shapefn_grad(x, jnp.array([-1, 1]))
    assert jnp.allclose(calc_grad, true_grad)


def test_shapefn_2d_scalar():
    point = (0.5, 0.5)
    sf = Linear4NodeQuad()
    true_sf = jnp.array([[0.0625, 0.1875, 0.5625, 0.1875]])
    calc_sf = sf.shapefn(*point)
    print(calc_sf)
    assert jnp.alltrue(calc_sf == true_sf)


def test_shapefn_2d_scalar_grad():
    x = (0.5, 0.5)
    sf = Linear4NodeQuad()
    true_grad = jnp.eye(2)
    calc_sfgrad = sf.shapefn_grad(
        x, jnp.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    )
    assert jnp.all(calc_sfgrad == true_grad)


def test_shapefn_2d_vector_grad():
    x = jnp.array([[0.5, 0.5], [0.5, 0.5]])
    sf = Linear4NodeQuad()
    true_grad = jnp.eye(2)
    calc_sfgrad = sf.shapefn_grad(
        x, jnp.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    )
    assert jnp.all(calc_sfgrad == true_grad)
