from collections import namedtuple
from jax.tree_util import register_pytree_node

NodalForce = namedtuple("NodalForce", ("node_ids", "function", "dir", "force"))
ParticleTraction = namedtuple(
    "ParticleTraction", ("pset", "pids", "function", "dir", "traction")
)
register_pytree_node(
    NodalForce,
    lambda xs: (tuple(xs), None),  # tell JAX how to unpack to an iterable
    lambda _, xs: NodalForce(*xs),  # tell JAX how to pack back into a NodalForce
)
register_pytree_node(
    ParticleTraction,
    lambda xs: (tuple(xs), None),  # tell JAX how to unpack to an iterable
    lambda _, xs: ParticleTraction(*xs),  # tell JAX how to pack back
)
