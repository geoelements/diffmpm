from typing import Optional, Sized, Tuple

import jax.numpy as jnp
from jax import jit, lax, vmap
from jax.tree_util import register_pytree_node_class, Partial
from jax.typing import ArrayLike

from diffmpm.element import _ElementsState
from diffmpm.materials import _Material

import chex


@chex.dataclass(frozen=True)
class _ParticlesState:
    ids: chex.ArrayDevice
    nparticles: int
    loc: chex.ArrayDevice
    material: _Material
    element_ids: chex.ArrayDevice
    mass: chex.ArrayDevice
    density: chex.ArrayDevice
    volume: chex.ArrayDevice
    size: chex.ArrayDevice
    velocity: chex.ArrayDevice
    acceleration: chex.ArrayDevice
    momentum: chex.ArrayDevice
    strain: chex.ArrayDevice
    stress: chex.ArrayDevice
    strain_rate: chex.ArrayDevice
    dstrain: chex.ArrayDevice
    f_ext: chex.ArrayDevice
    traction: chex.ArrayDevice
    reference_loc: chex.ArrayDevice
    dvolumetric_strain: chex.ArrayDevice
    volumetric_strain_centroid: chex.ArrayDevice
    state_vars: dict


def init_particle_state(
    loc: chex.ArrayDevice,
    material: _Material,
    element_ids: chex.ArrayDevice,
    init_vel: chex.ArrayDevice = 0,
):
    """Initialize a container of particles.

    Parameters
    ----------
    loc: ArrayLike
        Location of the particles. Expected shape (nparticles, 1, ndim)
    material: diffmpm.materials._Material
        Type of material for the set of particles.
    element_ids: ArrayLike
        The element ids that the particles belong to. This contains
        information that will make sense only with the information of
        the mesh that is being considered.
    initialized: bool
        `False` if particle property arrays like mass need to be initialized.
        If `True`, they are set to values from `data`.
    data: tuple
        Tuple of length 13 that sets arrays for mass, density, volume,
        velocity, acceleration, momentum, strain, stress, strain_rate,
        dstrain, f_ext, reference_loc and volumetric_strain_centroid.
    """
    loc = jnp.asarray(loc, dtype=jnp.float32)
    if loc.ndim != 3:
        raise ValueError(
            f"`loc` should be of size (nparticles, 1, ndim); " f"found {loc.shape}"
        )

    ids = jnp.arange(loc.shape[0])
    mass = jnp.ones((loc.shape[0], 1, 1))
    density = jnp.ones_like(mass) * material.density
    volume = jnp.ones_like(mass)
    size = jnp.zeros_like(loc)
    velocity = jnp.ones_like(loc) * init_vel
    acceleration = jnp.zeros_like(loc)
    momentum = jnp.zeros_like(loc)
    strain = jnp.zeros((loc.shape[0], 6, 1))
    stress = jnp.zeros((loc.shape[0], 6, 1))
    strain_rate = jnp.zeros((loc.shape[0], 6, 1))
    dstrain = jnp.zeros((loc.shape[0], 6, 1))
    f_ext = jnp.zeros_like(loc)
    traction = jnp.zeros_like(loc)
    reference_loc = jnp.zeros_like(loc)
    dvolumetric_strain = jnp.zeros((loc.shape[0], 1))
    volumetric_strain_centroid = jnp.zeros((loc.shape[0], 1))
    state_vars = {}
    if material.state_vars:
        state_vars = material.initialize_state_variables(loc.shape[0])
    return _ParticlesState(
        ids=ids,
        nparticles=loc.shape[0],
        loc=loc,
        material=material,
        element_ids=element_ids,
        mass=mass,
        density=density,
        volume=volume,
        size=size,
        velocity=velocity,
        acceleration=acceleration,
        momentum=momentum,
        strain=strain,
        stress=stress,
        strain_rate=strain_rate,
        dstrain=dstrain,
        f_ext=f_ext,
        traction=traction,
        reference_loc=reference_loc,
        dvolumetric_strain=dvolumetric_strain,
        volumetric_strain_centroid=volumetric_strain_centroid,
        state_vars=state_vars,
    )


# TODO: Can these methods just return the updated arrays to
# a single function which then generates the new state?


def set_mass_volume(state, m: ArrayLike) -> _ParticlesState:
    """Set particle mass.

    Parameters
    ----------
    m: float, array_like
        Mass to be set for particles. If scalar, mass for all
        particles is set to this value.
    """
    m = jnp.asarray(m)
    if jnp.isscalar(m):
        mass = jnp.ones_like(state.loc) * m
    elif m.shape == state.mass.shape:
        mass = m
    else:
        raise ValueError(
            f"Incompatible shapes. Expected {state.mass.shape}, " f"found {m.shape}."
        )
    volume = jnp.divide(mass, state.material.properties["density"])
    return state.replace(mass=mass, volume=volume)


def _compute_particle_volume(
    element_ids, total_elements, evol, pvol, psize, pmass, pdensity
):
    """Compute volume of all particles.

    Parameters
    ----------
    state:
        Current state
    elements: diffmpm._ElementState
        Elements that the particles are present in, and are used to
        compute the particles' volumes.
    total_elements: int
        Total elements present in `elements`.
    """
    particles_per_element = jnp.bincount(element_ids, length=total_elements)
    vol = (
        evol.squeeze((1, 2))[element_ids]  # type: ignore
        / particles_per_element[element_ids]
    )
    volume = pvol.at[:, 0, 0].set(vol)
    size = psize.at[:].set(volume ** (1 / psize.shape[-1]))
    mass = pmass.at[:, 0, 0].set(vol * pdensity.squeeze())
    return {"mass": mass, "size": size, "volume": volume}


def compute_volume(state, elements: _ElementsState, elementor, total_elements: int):
    """Compute volume of all particles.

    Parameters
    ----------
    state:
        Current state
    elements: diffmpm._ElementState
        Elements that the particles are present in, and are used to
        compute the particles' volumes.
    total_elements: int
        Total elements present in `elements`.
    """
    particles_per_element = jnp.bincount(
        state.element_ids, length=elementor.total_elements
    )
    vol = (
        elements.volume.squeeze((1, 2))[state.element_ids]  # type: ignore
        / particles_per_element[state.element_ids]
    )
    volume = state.volume.at[:, 0, 0].set(vol)
    size = state.size.at[:].set(volume ** (1 / state.size.shape[-1]))
    mass = state.mass.at[:, 0, 0].set(vol * state.density.squeeze())
    return state.replace(mass=mass, size=size, volume=volume)


def _get_natural_coords(particles, p_mapped_ids, eloc):
    r"""Update natural coordinates for the particles.

    Whenever the particles' physical coordinates change, their
    natural coordinates need to be updated. This function updates
    the natural coordinates of the particles based on the element
    a particle is a part of. The update formula is

    \[
        \xi = (2x - (x_1^e + x_2^e))  / (x_2^e - x_1^e)
    \]

    where \(x_i^e\) are the nodal coordinates of the element the
    particle is in. If a particle is not in any element
    (element_id = -1), its natural coordinate is set to 0.

    Parameters
    ----------
    elements: diffmpm.element._ElementState
        Elements based on which to update the natural coordinates
        of the particles.
    """
    t = eloc[p_mapped_ids.squeeze(-1)]
    xi_coords = (particles.loc - (t[:, 0, ...] + t[:, 2, ...]) / 2) * (
        2 / (t[:, 2, ...] - t[:, 0, ...])
    )
    return xi_coords


def update_natural_coords(state, elements: _ElementsState, elementor, *args):
    r"""Update natural coordinates for the particles.

    Whenever the particles' physical coordinates change, their
    natural coordinates need to be updated. This function updates
    the natural coordinates of the particles based on the element
    a particle is a part of. The update formula is

    \[
        \xi = (2x - (x_1^e + x_2^e))  / (x_2^e - x_1^e)
    \]

    where \(x_i^e\) are the nodal coordinates of the element the
    particle is in. If a particle is not in any element
    (element_id = -1), its natural coordinate is set to 0.

    Parameters
    ----------
    elements: diffmpm.element._ElementState
        Elements based on which to update the natural coordinates
        of the particles.
    """
    t = vmap(Partial(elementor.id_to_node_loc, elements))(state.element_ids)
    xi_coords = (state.loc - (t[:, 0, ...] + t[:, 2, ...]) / 2) * (
        2 / (t[:, 2, ...] - t[:, 0, ...])
    )
    return state.replace(reference_loc=xi_coords)


def update_position_velocity(
    state, elements: _ElementsState, elementor, dt: float, velocity_update: bool, *args
):
    """Transfer nodal velocity to particles and update particle position.

    The velocity is calculated based on the total force at nodes.

    Parameters
    ----------
    elements: diffmpm.element._ElementState
        Elements whose nodes are used to transfer the velocity.
    dt: float
        Timestep.
    velocity_update: bool
        If True, velocity is directly used as nodal velocity, else
        velocity is calculated is interpolated nodal acceleration
        multiplied by dt. Default is False.
    """
    mapped_positions = elementor.shapefn(state.reference_loc)
    mapped_ids = vmap(Partial(elementor.id_to_node_ids, elements.nelements[0]))(
        state.element_ids
    ).squeeze(-1)
    nodal_velocity = jnp.sum(
        mapped_positions * elements.nodes.velocity[mapped_ids], axis=1
    )
    nodal_acceleration = jnp.sum(
        mapped_positions * elements.nodes.acceleration[mapped_ids],
        axis=1,
    )
    velocity = state.velocity.at[:].set(
        lax.cond(
            velocity_update,
            lambda sv, nv, na, t: nv,
            lambda sv, nv, na, t: sv + na * t,
            state.velocity,
            nodal_velocity,
            nodal_acceleration,
            dt,
        )
    )
    loc = state.loc.at[:].add(nodal_velocity * dt)
    momentum = state.momentum.at[:].set(state.mass * state.velocity)
    return state.replace(velocity=velocity, loc=loc, momentum=momentum)


def _update_particle_position_velocity(
    el_type,
    ploc,
    pvel,
    pmom,
    pmass,
    pxi,
    mapped_node_ids,
    nvel,
    nacc,
    velocity_update,
    dt,
):
    """Transfer nodal velocity to particles and update particle position.

    The velocity is calculated based on the total force at nodes.

    Parameters
    ----------
    elements: diffmpm.element._ElementState
        Elements whose nodes are used to transfer the velocity.
    dt: float
        Timestep.
    velocity_update: bool
        If True, velocity is directly used as nodal velocity, else
        velocity is calculated is interpolated nodal acceleration
        multiplied by dt. Default is False.
    """
    mapped_positions = el_type._shapefn(pxi)
    mapped_ids = mapped_node_ids.squeeze(-1)
    nodal_velocity = jnp.sum(mapped_positions * nvel[mapped_ids], axis=1)
    nodal_acceleration = jnp.sum(
        mapped_positions * nacc[mapped_ids],
        axis=1,
    )
    velocity = lax.cond(
        velocity_update,
        lambda sv, nv, na, t: nv,
        lambda sv, nv, na, t: sv + na * t,
        pvel,
        nodal_velocity,
        nodal_acceleration,
        dt,
    )
    loc = ploc.at[:].add(nodal_velocity * dt)
    momentum = pmass * pvel
    return {"velocity": velocity, "loc": loc, "momentum": momentum}


def _compute_strain_rate(mapped_vel, pids, dn_dx: ArrayLike):
    """Compute the strain rate for particles.

    Parameters
    ----------
    dn_dx: ArrayLike
        The gradient of the shape function. Expected shape
        `(nparticles, 1, ndim)`
    elements: diffmpm.element._ElementState
        Elements whose nodes are used to calculate the strain rate.
    """
    dn_dx = jnp.asarray(dn_dx)
    strain_rate = jnp.zeros((dn_dx.shape[0], 6, 1))  # (nparticles, 6, 1)
    temp = mapped_vel.squeeze(2)

    @jit
    def _step(pid, args):
        dndx, nvel, strain_rate = args
        matmul = dndx[pid].T @ nvel[pid]
        strain_rate = strain_rate.at[pid, 0].add(matmul[0, 0])
        strain_rate = strain_rate.at[pid, 1].add(matmul[1, 1])
        strain_rate = strain_rate.at[pid, 3].add(matmul[0, 1] + matmul[1, 0])
        return dndx, nvel, strain_rate

    def _scan_step(carry, pid):
        dndx, nvel, strain_rate = carry
        matmul = dndx[pid].T @ nvel[pid]
        strain_rate = strain_rate.at[pid, 0].add(matmul[0, 0])
        strain_rate = strain_rate.at[pid, 1].add(matmul[1, 1])
        strain_rate = strain_rate.at[pid, 3].add(matmul[0, 1] + matmul[1, 0])
        return (dndx, nvel, strain_rate), pid

    args = (dn_dx, temp, strain_rate)
    # _, _, strain_rate = lax.fori_loop(0, nparticles, _step, args)
    final_carry, _ = lax.scan(_scan_step, args, pids)
    _, _, strain_rate = final_carry
    strain_rate = jnp.where(jnp.abs(strain_rate) < 1e-12, 0, strain_rate)
    return strain_rate


def _compute_strain(
    pstrain,
    pxi,
    ploc,
    pvolumetric_strain_centroid,
    pids,
    mapped_node_ids,
    nloc,
    nvel,
    el_type,
    dt,
):
    """Compute the strain on all particles.

    This is done by first calculating the strain rate for the particles
    and then calculating strain as `strain += strain rate * dt`.

    Parameters
    ----------
    elements: diffmpm.element._ElementState
        Elements whose nodes are used to calculate the strain.
    dt : float
        Timestep.
    """
    mapped_nodes = mapped_node_ids.squeeze(-1)
    mapped_coords = nloc[mapped_nodes]
    mapped_vel = nvel[mapped_nodes]
    dn_dx_ = vmap(el_type._shapefn_grad)(pxi[:, jnp.newaxis, ...], mapped_coords)
    new_strain_rate = _compute_strain_rate(mapped_vel, pids, dn_dx_)
    new_dstrain = new_strain_rate * dt

    new_strain = pstrain + new_dstrain
    centroids = jnp.zeros_like(ploc)
    dn_dx_centroid_ = vmap(jit(el_type._shapefn_grad))(
        centroids[:, jnp.newaxis, ...], mapped_coords
    )
    strain_rate_centroid = _compute_strain_rate(
        mapped_vel,
        pids,
        dn_dx_centroid_,
    )
    ndim = ploc.shape[-1]
    new_dvolumetric_strain = dt * strain_rate_centroid[:, :ndim].sum(axis=1)
    new_volumetric_strain_centroid = (
        pvolumetric_strain_centroid + new_dvolumetric_strain
    )
    return {
        "strain_rate": new_strain_rate,
        "dstrain": new_dstrain,
        "strain": new_strain,
        "dvolumetric_strain": new_dvolumetric_strain,
        "volumetric_strain_centroid": new_volumetric_strain_centroid,
    }


def compute_strain(state, elements: _ElementsState, elementor, dt: float, *args):
    """Compute the strain on all particles.

    This is done by first calculating the strain rate for the particles
    and then calculating strain as `strain += strain rate * dt`.

    Parameters
    ----------
    elements: diffmpm.element._ElementState
        Elements whose nodes are used to calculate the strain.
    dt : float
        Timestep.
    """
    # breakpoint()
    mapped_coords = vmap(Partial(elementor.id_to_node_loc, elements))(
        state.element_ids
    ).squeeze(2)
    mapped_vel = vmap(Partial(elementor.id_to_node_vel, elements))(state.element_ids)
    dn_dx_ = vmap(jit(elementor.shapefn_grad))(
        state.reference_loc[:, jnp.newaxis, ...], mapped_coords
    )
    # strain_rate = _compute_strain_rate(state, dn_dx_, elements, elementor)
    strain_rate = _compute_strain_rate(mapped_vel, state.nparticles, dn_dx_)
    dstrain = state.dstrain.at[:].set(strain_rate * dt)

    strain = state.strain.at[:].add(dstrain)
    centroids = jnp.zeros_like(state.loc)
    dn_dx_centroid_ = vmap(jit(elementor.shapefn_grad))(
        centroids[:, jnp.newaxis, ...], mapped_coords
    )
    # strain_rate_centroid = _compute_strain_rate(
    #     state, dn_dx_centroid_, elements, elementor
    # )
    strain_rate_centroid = _compute_strain_rate(
        mapped_vel, state.nparticles, dn_dx_centroid_
    )
    ndim = state.loc.shape[-1]
    dvolumetric_strain = dt * strain_rate_centroid[:, :ndim].sum(axis=1)
    volumetric_strain_centroid = state.volumetric_strain_centroid.at[:].add(
        dvolumetric_strain
    )
    return state.replace(
        strain_rate=strain_rate,
        dstrain=dstrain,
        strain=strain,
        dvolumetric_strain=dvolumetric_strain,
        volumetric_strain_centroid=volumetric_strain_centroid,
    )


def compute_stress(state, *args):
    """Compute the strain on all particles.

    This calculation is governed by the material of the
    particles. The stress calculated by the material is then
    added to the particles current stress values.
    """
    stress = state.stress.at[:].add(state.material.compute_stress(state))
    return state.replace(stress=stress)


def _compute_stress(stress, strain, dstrain, material, *args):
    """Compute the strain on all particles.

    This calculation is governed by the material of the
    particles. The stress calculated by the material is then
    added to the particles current stress values.
    """
    new_stress = stress + material.compute_stress(strain, dstrain)
    return new_stress


def update_volume(state, *args):
    """Update volume based on central strain rate."""
    volume = state.volume.at[:, 0, :].multiply(1 + state.dvolumetric_strain)
    density = state.density.at[:, 0, :].divide(1 + state.dvolumetric_strain)
    return state.replace(volume=volume, density=density)


def _update_particle_volume(pvol, pdensity, pdvolumetric_strain):
    """Update volume based on central strain rate."""
    new_volume = pvol.at[:, 0, :].multiply(1 + pdvolumetric_strain)
    new_density = pdensity.at[:, 0, :].divide(1 + pdvolumetric_strain)
    return new_volume, new_density


def assign_traction(state, pids: ArrayLike, dir: int, traction_: float):
    """Assign traction to particles.

    Parameters
    ----------
    pids: ArrayLike
        IDs of the particles to which traction should be applied.
    dir: int
        The direction in which traction should be applied.
    traction_: float
        Traction value to be applied in the direction.
    """
    traction = state.traction.at[pids, 0, dir].add(
        traction_ * state.volume[pids, 0, 0] / state.size[pids, 0, dir]
    )
    return traction


def _assign_traction(
    ptraction,
    pvol,
    psize,
    pids: ArrayLike,
    dir: int,
    traction_val_: float,
):
    """Assign traction to particles.

    Parameters
    ----------
    pids: ArrayLike
        IDs of the particles to which traction should be applied.
    dir: int
        The direction in which traction should be applied.
    traction_: float
        Traction value to be applied in the direction.
    """
    traction = ptraction.at[pids, 0, dir].add(
        traction_val_ * pvol[pids, 0, 0] / psize[pids, 0, dir]
    )
    return traction


def _zero_traction(traction):
    """Set all traction values to 0."""
    traction = jnp.zeros_like(traction)
    return traction


def zero_traction(state, *args):
    """Set all traction values to 0."""
    traction = state.traction.at[:].set(0)
    return state.replace(traction=traction)
