from collections import namedtuple

NodalForce = namedtuple("NodalForce", ("node_ids", "function", "dir", "force"))
ParticleTraction = namedtuple(
    "ParticleTraction", ("pset", "function", "dir", "traction")
)
