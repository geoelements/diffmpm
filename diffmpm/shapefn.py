import jax.numpy as jnp
from jax import jacobian, vmap, lax
from jax.tree_util import register_pytree_node_class
from abc import ABC, abstractmethod


class ShapeFn(ABC):
    """
    Baseclass to implement Shape functions.

    These define the required methods any class implementing a Shape
    function should have to interact correctly with the rest of the
    `diffmpm` library.
    """

    @abstractmethod
    def tree_flatten(self):
        """
        Define how to flatten a class object.

        This is needed to use a class object as a PyTree node.
        """
        ...

    @classmethod
    def tree_unflatten(self):
        """
        Define how to unflatten a class object.

        This is needed to use a class object as a PyTree node.
        """
        ...

    @abstractmethod
    def shapefn(self, xi):
        """
        Return value of the shape function.

        Based on the dimension provided to the constructor, this returns
        the value of the shape function in the natural coordinates.
        """
        ...

    def _shapefn_natural_grad(self, xi):
        """
        Calculate the gradient of shape function.

        This calculation is done in the natural coordinates.
        """
        return xi

    @abstractmethod
    def shapefn_grad(self, x, coords):
        """Gradient of shape function in physical coordinates."""
        ...


@register_pytree_node_class
class Linear1DShapeFn(ShapeFn):
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

    def tree_flatten(self):
        return ((self.dim,), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(*children)

    def __repr__(self):
        return f"ShapeFn(dim={self.dim})"

    def shapefn(self, xi):
        """
        Return value of the shape function.

        Based on the dimension provided to the constructor, this returns
        the value of the shape function in the natural coordinates.

        Arguments
        ---------
        xi : float, array_like
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
            result = jnp.array([0.5 * (1 - xi), 0.5 * (1 + xi)]).T
        return result

    def _shapefn_natural_grad(self, xi):
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
        if jnp.isscalar(xi):
            result = jacobian(self.shapefn)(xi)
        else:
            result = vmap(jacobian(self.shapefn))(xi.reshape(-1, 1)).squeeze()

        # TODO: The following code tries to evaluate vmap even if
        # the predicate condition is true, not sure why.
        # result = lax.cond(
        #     jnp.isscalar(x),
        #     jacobian(self.shapefn),
        #     vmap(jacobian(self.shapefn)),
        #     xi
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
        if self.dim == 1:
            length = abs(coords[1] - coords[0])
            result = self._shapefn_natural_grad(x) * 2 / length
        return result


@register_pytree_node_class
class Linear4NodeQuad(ShapeFn):
    """
    Define a 2D shape function for an element.

    Attributes
    ----------
    dim : int
        Dimension of the shape function.
    """

    def __init__(self):
        """
        Construct Shape functions for the given dimension.

        Arguments
        ---------
        dim : int
            Dimension of the shape function to be used.
        """
        self.dim = 2
        return

    def tree_flatten(self):
        return ((self.dim,), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(*children)

    def __repr__(self):
        return f"{self.__class__.__name__}(dim={self.dim})"

    def shapefn(self, xi, yi):
        """
        Return value of the shape function.

        Return shape function for a 2D Quadrilateral Linear Shape function.

        Arguments
        ---------
        xi : float, array_like
            x-coordinates in the natural coordinate system to evaluate the
        function at.

        yi : float, array_like
            y-coordinates in the natural coordinate system to evaluate the
        function at.
        Returns
        -------
        array_like
            Evaluated shape function values. The shape of the returned
        array will depend on the input shape.
        """
        result_0 = 0.25 * (1 - xi) * (1 - yi)
        result_1 = 0.25 * (1 + xi) * (1 - yi)
        result_2 = 0.25 * (1 + xi) * (1 + yi)
        result_3 = 0.25 * (1 - xi) * (1 + yi)
        return jnp.asarray([result_0, result_1, result_2, result_3])

    def _shapefn_natural_grad(self, xi, yi):
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
        the returned array will depend on the input shape.
        """
        if jnp.isscalar(xi):
            result = jnp.asarray(jacobian(self.shapefn, argnums=(0, 1))(xi, yi))
        else:
            result = vmap(jacobian(self.shapefn, argnums=(0, 1)))(
                xi.reshape(-1, 1), yi.reshape(-1, 1)
            )
            result = jnp.asarray(result).squeeze().transpose(1, 0, 2)

        # TODO: The following code tries to evaluate vmap even if
        # the predicate condition is true, not sure why.
        # result = lax.cond(
        #     jnp.isscalar(x),
        #     jacobian(self.shapefn),
        #     vmap(jacobian(self.shapefn)),
        #     xi
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
        natural_grad = self._shapefn_natural_grad(*x)
        result = natural_grad @ coords
        return result
