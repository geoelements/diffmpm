from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class Material:
    """
    Base material class.
    """

    def __init__(self, E, density):
        """
        Initialize material properties.

        Arguments
        ---------
        E : float
            Young's modulus of the material.
        density : float
            Density of the material.
        """
        self.E = E
        self.density = density

    def tree_flatten(self):
        return (tuple(), (self.E, self.density))

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        return cls(*aux_data)

    def __repr__(self):
        return f"Material(E={self.E}, density={self.density})"
