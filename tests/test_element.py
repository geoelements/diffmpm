import jax.numpy as jnp
from diffmpm.element import Quadrilateral4Node


class TestLinear1D:
    pass


class TestQuadrilateral4Node:
    elements = Quadrilateral4Node((1, 1), 1, (1.0, 1.0), [])

    def test_shapefn(self):
        particle_coord = jnp.array([[0, 0]]).reshape(-1, 1, 2)
        sfn = self.elements.shapefn(particle_coord)
        assert jnp.allclose(
            sfn, jnp.array([0.25, 0.25, 0.25, 0.25]).reshape(1, 4, 1, 1)
        )

        particle_coord = jnp.array([[-1, -1]]).reshape(-1, 1, 2)
        sfn = self.elements.shapefn(particle_coord)
        assert jnp.allclose(sfn, jnp.array([1, 0, 0, 0]).reshape(1, 4, 1, 1))

        particle_coord = jnp.array([[1, 1]]).reshape(-1, 1, 2)
        sfn = self.elements.shapefn(particle_coord)
        assert jnp.allclose(sfn, jnp.array([0, 0, 1, 0]).reshape(1, 4, 1, 1))

    def test_shapefn_natural_grad(self):
        particle_coord = jnp.array([[0, 0]]).reshape(-1, 1, 2)
        sfn_grad = self.elements._shapefn_natural_grad(particle_coord)
        assert jnp.allclose(
            sfn_grad,
            jnp.array(
                [[-0.25, -0.25], [0.25, -0.25], [0.25, 0.25], [-0.25, 0.25]]
            ).reshape(-1, 4, 2),
        )

        particle_coord = jnp.array([[-1, -1]]).reshape(-1, 1, 2)
        sfn_grad = self.elements._shapefn_natural_grad(particle_coord)
        assert jnp.allclose(
            sfn_grad,
            jnp.array([[-0.5, -0.5], [0.5, 0.0], [0.0, 0.0], [0.0, 0.5]]).reshape(
                -1, 4, 2
            ),
        )

        particle_coord = jnp.array([[1, 1]]).reshape(-1, 1, 2)
        sfn_grad = self.elements._shapefn_natural_grad(particle_coord)
        assert jnp.allclose(
            sfn_grad,
            jnp.array([[0.0, 0.0], [0.0, -0.5], [0.5, 0.5], [-0.5, 0.0]]).reshape(
                -1, 4, 2
            ),
        )

    def test_shapefn_grad(self):
        particle_coord = jnp.array([[0.5, 0.5]]).reshape(-1, 1, 2)
        nodal_coords = jnp.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        ).reshape(-1, 4, 2)
        sfn_natural_grad = self.elements._shapefn_natural_grad(particle_coord) * 2
        sfn_grad = self.elements.shapefn_grad(particle_coord, nodal_coords)
        assert jnp.allclose(sfn_grad, sfn_natural_grad)

        particle_coord = jnp.array([[-0.5, -0.5]]).reshape(-1, 1, 2)
        nodal_coords = jnp.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        ).reshape(-1, 4, 2)
        sfn_natural_grad = self.elements._shapefn_natural_grad(particle_coord) * 2
        sfn_grad = self.elements.shapefn_grad(particle_coord, nodal_coords)
        assert jnp.allclose(sfn_grad, sfn_natural_grad)
