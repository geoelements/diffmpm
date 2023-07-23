import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from ._base import _Material

import chex


@chex.dataclass()
class _LinearElasticState:
    id: int
    state_vars: tuple
    density: float
    youngs_modulus: float
    poisson_ratio: float
    bulk_modulus: float
    pwave_velocity: float
    swave_velocity: float
    de: chex.ArrayDevice

    def compute_stress(self, state):
        """Compute material stress."""
        dstress = self.de @ state.dstrain
        return dstress


def init_linear_elastic(material_properties):
    """Create a Linear Elastic material.

    Parameters
    ----------
    material_properties: dict
        Dictionary with material properties. For linear elastic
    materials, 'density' and 'youngs_modulus' are required keys.
    """
    state_vars = ()
    youngs_modulus = material_properties["youngs_modulus"]
    poisson_ratio = material_properties["poisson_ratio"]
    density = material_properties["density"]
    bulk_modulus = youngs_modulus / (3 * (1 - 2 * poisson_ratio))
    constrained_modulus = (
        youngs_modulus
        * (1 - poisson_ratio)
        / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
    )
    shear_modulus = youngs_modulus / (2 * (1 + poisson_ratio))
    # Wave velocities
    vp = jnp.sqrt(constrained_modulus / density)
    vs = jnp.sqrt(shear_modulus / density)
    properties = {
        **material_properties,
        "bulk_modulus": bulk_modulus,
        "pwave_velocity": vp,
        "swave_velocity": vs,
    }
    G = youngs_modulus / (2 * (1 + poisson_ratio))

    a1 = bulk_modulus + (4 * G / 3)
    a2 = bulk_modulus - (2 * G / 3)

    de = jnp.array(
        [
            [a1, a2, a2, 0, 0, 0],
            [a2, a1, a2, 0, 0, 0],
            [a2, a2, a1, 0, 0, 0],
            [0, 0, 0, G, 0, 0],
            [0, 0, 0, 0, G, 0],
            [0, 0, 0, 0, 0, G],
        ]
    )
    return _LinearElasticState(**properties, de=de, state_vars=state_vars)


# @register_pytree_node_class
# class LinearElastic(_Material):
#     """Linear Elastic Material."""

#     _props = ("density", "youngs_modulus", "poisson_ratio")
#     state_vars = ()

#     def __init__(self, material_properties):
#         """Create a Linear Elastic material.

#         Parameters
#         ----------
#         material_properties: dict
#             Dictionary with material properties. For linear elastic
#         materials, 'density' and 'youngs_modulus' are required keys.
#         """
#         self.validate_props(material_properties)
#         youngs_modulus = material_properties["youngs_modulus"]
#         poisson_ratio = material_properties["poisson_ratio"]
#         density = material_properties["density"]
#         bulk_modulus = youngs_modulus / (3 * (1 - 2 * poisson_ratio))
#         constrained_modulus = (
#             youngs_modulus
#             * (1 - poisson_ratio)
#             / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
#         )
#         shear_modulus = youngs_modulus / (2 * (1 + poisson_ratio))
#         # Wave velocities
#         vp = jnp.sqrt(constrained_modulus / density)
#         vs = jnp.sqrt(shear_modulus / density)
#         self.properties = {
#             **material_properties,
#             "bulk_modulus": bulk_modulus,
#             "pwave_velocity": vp,
#             "swave_velocity": vs,
#         }
#         self._compute_elastic_tensor()

#     def __repr__(self):
#         return f"LinearElastic(props={self.properties})"

#     def _compute_elastic_tensor(self):
#         G = self.properties["youngs_modulus"] / (
#             2 * (1 + self.properties["poisson_ratio"])
#         )

#         a1 = self.properties["bulk_modulus"] + (4 * G / 3)
#         a2 = self.properties["bulk_modulus"] - (2 * G / 3)

#         self.de = jnp.array(
#             [
#                 [a1, a2, a2, 0, 0, 0],
#                 [a2, a1, a2, 0, 0, 0],
#                 [a2, a2, a1, 0, 0, 0],
#                 [0, 0, 0, G, 0, 0],
#                 [0, 0, 0, 0, G, 0],
#                 [0, 0, 0, 0, 0, G],
#             ]
#         )

#     def compute_stress(self, particles):
#         """Compute material stress."""
#         dstress = self.de @ particles.dstrain
#         return dstress
