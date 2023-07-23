import jax.numpy as jnp

import chex


@chex.dataclass(frozen=True)
class _NodesState:
    nnodes: int
    loc: chex.ArrayDevice
    velocity: chex.ArrayDevice
    acceleration: chex.ArrayDevice
    mass: chex.ArrayDevice
    momentum: chex.ArrayDevice
    f_int: chex.ArrayDevice
    f_ext: chex.ArrayDevice
    f_damp: chex.ArrayDevice


def init_node_state(
    nnodes: int,
    loc: chex.ArrayDevice,
):
    """Initialize container for Nodes.

    Parameters
    ----------
    nnodes : int
        Number of nodes stored.
    loc : ArrayLike
        Locations of all the nodes. Expected shape (nnodes, 1, ndim)
    initialized: bool
        `False` if node property arrays like mass need to be initialized.
        If `True`, they are set to values from `data`.
    data: tuple
        Tuple of length 7 that sets arrays for mass, density, volume,
        and forces. Mainly used by JAX while unflattening.
    """
    loc = jnp.asarray(loc, dtype=jnp.float32)
    if loc.ndim != 3 or nnodes != loc.shape[0]:
        raise ValueError(
            f"`loc` should be of size (nnodes, 1, ndim); found {loc.shape}"
        )

    velocity = jnp.zeros_like(loc, dtype=jnp.float32)
    acceleration = jnp.zeros_like(loc, dtype=jnp.float32)
    mass = jnp.zeros((loc.shape[0], 1, 1), dtype=jnp.float32)
    momentum = jnp.zeros_like(loc, dtype=jnp.float32)
    f_int = jnp.zeros_like(loc, dtype=jnp.float32)
    f_ext = jnp.zeros_like(loc, dtype=jnp.float32)
    f_damp = jnp.zeros_like(loc, dtype=jnp.float32)
    return _NodesState(
        nnodes=nnodes,
        loc=loc,
        velocity=velocity,
        acceleration=acceleration,
        mass=mass,
        momentum=momentum,
        f_int=f_int,
        f_ext=f_ext,
        f_damp=f_damp,
    )


def reset_node_state(state: _NodesState):
    mass = state.mass.at[:].set(0)
    momentum = state.momentum.at[:].set(0)
    f_int = state.f_int.at[:].set(0)
    f_ext = state.f_ext.at[:].set(0)
    f_damp = state.f_damp.at[:].set(0)
    return state.replace(
        mass=mass,
        momentum=momentum,
        f_int=f_int,
        f_ext=f_ext,
        f_damp=f_damp,
    )
