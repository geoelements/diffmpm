from typing import Annotated, NamedTuple, get_type_hints

from jax import Array
from jax.tree_util import register_pytree_node

from diffmpm.functions import Function


class NodalForce(NamedTuple):
    """Nodal Force being applied constantly on a set of nodes."""

    node_ids: Annotated[Array, "Array of Node IDs to which force is applied."]
    function: Annotated[
        Function,
        "Mathematical function that governs time-varying changes in the force.",
    ]
    dir: Annotated[int, "Direction in which force is applied."]
    force: Annotated[float, "Amount of force to be applied."]


nfhints = get_type_hints(NodalForce, include_extras=True)
for attr in nfhints:
    getattr(NodalForce, attr).__doc__ = "".join(nfhints[attr].__metadata__)


class ParticleTraction(NamedTuple):
    """Traction being applied on a set of particles."""

    pset: Annotated[
        int, "The particle set in which traction is applied to the particles."
    ]
    pids: Annotated[
        Array,
        "Array of Particle IDs to which traction is applied inside the particle set.",
    ]
    function: Annotated[
        Function,
        "Mathematical function that governs time-varying changes in the traction.",
    ]
    dir: Annotated[int, "Direction in which traction is applied."]
    traction: Annotated[float, "Amount of traction to be applied."]


pthints = get_type_hints(ParticleTraction, include_extras=True)
for attr in pthints:
    getattr(ParticleTraction, attr).__doc__ = "".join(pthints[attr].__metadata__)

register_pytree_node(
    NodalForce,
    # tell JAX how to unpack to an iterable
    lambda xs: (tuple(xs), None),  # type: ignore
    # tell JAX how to pack back into a NodalForce
    lambda _, xs: NodalForce(*xs),  # type: ignore
)
register_pytree_node(
    ParticleTraction,
    # tell JAX how to unpack to an iterable
    lambda xs: (tuple(xs), None),  # type: ignore
    # tell JAX how to pack back
    lambda _, xs: ParticleTraction(*xs),  # type: ignore
)
