from jax.tree_util import register_pytree_node_class

from ._base import _Material


@register_pytree_node_class
class SimpleMaterial(_Material):
    _props = ("E", "density")

    def __init__(self, material_properties):
        self.validate_props(material_properties)
        self.properties = material_properties

    def __repr__(self):
        return f"SimpleMaterial(props={self.properties})"

    def compute_stress(self, dstrain):
        return dstrain * self.properties["E"]
