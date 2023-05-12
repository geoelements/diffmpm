from jax.tree_util import register_pytree_node_class
import abc


class Material(abc.ABC):
    """Base material class."""

    def __init__(self, material_properties):
        """
        Initialize material properties.

        Arguments
        ---------
        material_properties: dict
            A key-value map for various material properties.
        """
        self.material_properties = material_properties

    @abc.abstractmethod
    def tree_flatten(self):
        """Flatten this class as PyTree Node."""
        ...

    @classmethod
    @abc.abstractmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflatten this class as PyTree Node."""
        ...

    @abc.abstractmethod
    def __repr__(self):
        """Repr for Material class."""
        ...


@register_pytree_node_class
class LinearElastic(Material):
    """Linear Elastic Material."""

    _props = ("density", "youngs_modulus")

    def __init__(self, material_properties):
        """
        Create a Linear Elastic material.

        Arguments
        ---------
        material_properties: dict
            Dictionary with material properties. For linear elastic
        materials, 'density' and 'youngs_modulus' are required keys.
        """
        for key in self._props:
            if key not in material_properties:
                raise KeyError(
                    f"'{key}' should be present in `material_properties` "
                    f"for LinearElastic materials."
                )

        self.material_properties = material_properties
