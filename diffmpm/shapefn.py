import jax.numpy as jnp
from jax import jacobian, vmap, lax


class ShapeFn:
    """
    Define the shape function of an element.

    Based on the dimension, the shape function returns the required
    mapped nodal values.

    Attributes
    ----------
    dim : int
        Dimension of the shape function.
    """

    def __init__(self, dim=1):
        """
        Construct Shape functions for the given dimension.

        Arguments
        ---------
        dim : int
            Dimension of the shape function to be used.
        """
        self.dim = dim
        return

    def shapefn(self, x):
        """
        Return value of the shape function.

        Based on the dimension provided to the constructor, this returns
        the value of the shape function in the natural coordinates.

        Arguments
        ---------
        x : float, array_like
            Locations in natural coordinates to evaluate the function at.

        Returns
        -------
        array_like
            Evaluated shape function values. The shape of the returned
        array will depend on the input shape. For example, in the linear
        case, if the input is a scalar, the returned array will be of
        the shape (2,) but if the input is a vector then the output will
        be of the shape (len(x), 2).
        """
        if self.dim == 1:
            result = jnp.array([0.5 * (1 - x), 0.5 * (1 + x)])
        return result

    def _shapefn_natural_grad(self, x):
        """
        Calculate the gradient of shape function.

        This calculation is done in the natural coordinates.

        Arguments
        ---------
        x : float, array_like
            Locations in natural coordinates to evaluate the function at.

        Returns
        -------
        array_like
            Evaluated gradient values of the shape function. The shape of
        the returned array will depend on the input shape. For example,
        in the linear case, if the input is a scalar, the returned array
        will be of the shape (2,) but if the input is a vector then the
        output will be of the shape (len(x), 2).
        """
        if jnp.isscalar(x):
            result = jacobian(self.shapefn)(x)
        else:
            result = vmap(jacobian(self.shapefn))(x)

        # TODO: The following code tries to evaluate vmap even if
        # the predicate condition is true, not sure why.
        # result = lax.cond(
        #     jnp.isscalar(x),
        #     jacobian(self.shapefn),
        #     vmap(jacobian(self.shapefn)),
        #     x
        # )
        return result

    def shapefn_grad(self, x, coords):
        """
        Gradient of shape function in physical coordinates.

        Arguments
        ---------
        x : float, array_like
            Locations to evaluate in natural coordinates.
        coords : array_like
            Nodal coordinates to transform by.

        Returns
        -------
        array_like
            Gradient of the shape function in physical coordinates at `x`
        """
        length = abs(coords[1] - coords[0])
        result = self._shapefn_natural_grad(x) * 2 / length
        return result
