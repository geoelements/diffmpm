from jax.tree_util import register_pytree_node_class

from ._base import _Material

import chex


@chex.dataclass()
class _SimpleMaterialState:
    id: int
    E: float
    density: float
    state_vars: ()

    def compute_stress(self, strain, dstrain):
        return dstrain * self.E


def init_simple(material_properties):
    return _SimpleMaterialState(**material_properties, state_vars=())


@register_pytree_node_class
class SimpleMaterial(_Material):
    _props = ("E", "density")
    state_vars = ()

    def __init__(self, material_properties):
        self.validate_props(material_properties)
        self.properties = material_properties

    def __repr__(self):
        return f"SimpleMaterial(props={self.properties})"

    def compute_stress(self, strain, dstrain):
        return dstrain * self.properties["E"]
