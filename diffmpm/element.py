from __future__ import annotations

import abc
import itertools
from functools import partial
from typing import TYPE_CHECKING, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from diffmpm.particle import _ParticlesState

import jax.numpy as jnp
from jax import Array, jacobian, jit, lax, tree_util, vmap
from jax.tree_util import register_pytree_node_class, tree_map, tree_reduce, Partial
from jax.typing import ArrayLike

from diffmpm.constraint import Constraint
from diffmpm.forces import NodalForce
from diffmpm.node import _NodesState, init_node_state
import chex


@chex.dataclass()
class _ElementsState:
    nodes: _NodesState
    total_elements: int
    volume: chex.ArrayDevice
    constraints: Sequence[Tuple[ArrayLike, Constraint]]
    concentrated_nodal_forces: Sequence[NodalForce]


@chex.dataclass()
class Quad4NState(_ElementsState):
    nelements: chex.ArrayDevice
    el_len: chex.ArrayDevice


@chex.dataclass()
class Quad4N:
    total_elements: int

    def init_state(
        self,
        nelements: int,
        total_elements: int,
        el_len: float,
        constraints: Sequence[Tuple[ArrayLike, Constraint]],
        nodes: Optional[_NodesState] = None,
        concentrated_nodal_forces: Sequence = [],
        initialized: Optional[bool] = None,
        volume: Optional[ArrayLike] = None,
    ) -> Quad4NState:
        """Initialize Linear1D.

        Parameters
        ----------
        nelements : int
            Number of elements.
        total_elements : int
            Total number of elements (product of all elements of `nelements`)
        el_len : float
            Length of each element.
        constraints: list
            A list of constraints where each element is a tuple of
            type `(node_ids, diffmpm.Constraint)`. Here, `node_ids`
            correspond to the node IDs where `diffmpm.Constraint`
            should be applied.
        nodes : Nodes, Optional
            Nodes in the element object.
        concentrated_nodal_forces: list
            A list of `diffmpm.forces.NodalForce`s that are to be
            applied.
        initialized: bool, None
            `True` if the class has been initialized, `None` if not.
            This is required like this for using JAX flattening.
        volume: ArrayLike
            Volume of the elements.
        """
        nelements = jnp.asarray(nelements)
        el_len = jnp.asarray(el_len)

        total_nodes = jnp.prod(nelements + 1)
        coords = jnp.asarray(
            list(
                itertools.product(
                    jnp.arange(nelements[1] + 1),
                    jnp.arange(nelements[0] + 1),
                )
            )
        )
        node_locations = (jnp.asarray([coords[:, 1], coords[:, 0]]).T * el_len).reshape(
            -1, 1, 2
        )
        nodes = init_node_state(int(total_nodes), node_locations)

        volume = jnp.ones((total_elements, 1, 1))
        return Quad4NState(
            nodes=nodes,
            total_elements=total_elements,
            concentrated_nodal_forces=concentrated_nodal_forces,
            volume=volume,
            constraints=constraints,
            nelements=nelements,
            el_len=el_len,
        )

    def id_to_node_loc(self, elements: _ElementState, id: ArrayLike) -> Array:
        """Node locations corresponding to element `id`.

        Parameters
        ----------
        id : int
            Element ID.

        Returns
        -------
        ArrayLike
            Nodal locations for the element. Shape of returned
            array is `(nodes_in_element, 1, ndim)`
        """
        node_ids = self.id_to_node_ids(elements.nelements[0], id).squeeze()
        return elements.nodes.loc[node_ids]

    def id_to_node_vel(self, elements: _ElementState, id: ArrayLike) -> Array:
        """Node velocities corresponding to element `id`.

        Parameters
        ----------
        id : int
            Element ID.

        Returns
        -------
        ArrayLike
            Nodal velocities for the element. Shape of returned
            array is `(nodes_in_element, 1, ndim)`
        """
        node_ids = self.id_to_node_ids(elements.nelements[0], id).squeeze()
        return elements.nodes.velocity[node_ids]

    def id_to_node_ids(self, nelements_x, id: ArrayLike):
        """Node IDs corresponding to element `id`.

            3----2
            |    |
            0----1

        Node ids are returned in the order as shown in the figure.

        Parameters
        ----------
        id : int
            Element ID.

        Returns
        -------
        ArrayLike
            Nodal IDs of the element. Shape of returned
            array is (4, 1)
        """
        lower_left = (id // nelements_x) * (nelements_x + 1) + id % nelements_x
        result = jnp.asarray(
            [
                lower_left,
                lower_left + 1,
                lower_left + nelements_x + 2,
                lower_left + nelements_x + 1,
            ]
        )
        return result.reshape(4, 1)

    @classmethod
    def _get_mapped_nodes(cls, id, nelements_x):
        """Node IDs corresponding to element `id`.

            3----2
            |    |
            0----1

        Node ids are returned in the order as shown in the figure.

        Parameters
        ----------
        id : int
            Element ID.

        Returns
        -------
        ArrayLike
            Nodal IDs of the element. Shape of returned
            array is (4, 1)
        """
        lower_left = (id // nelements_x) * (nelements_x + 1) + id % nelements_x
        result = jnp.asarray(
            [
                lower_left,
                lower_left + 1,
                lower_left + nelements_x + 2,
                lower_left + nelements_x + 1,
            ]
        )
        return result.reshape(4, 1)

    def shapefn(self, xi: ArrayLike):
        """Evaluate linear shape function.

        Parameters
        ----------
        xi : float, array_like
            Locations of particles in natural coordinates to evaluate
            the function at. Expected shape is (npoints, 1, ndim)

        Returns
        -------
        array_like
            Evaluated shape function values. The shape of the returned
            array will depend on the input shape. For example, in the linear
            case, if the input is a scalar, the returned array will be of
            the shape `(1, 4, 1)` but if the input is a vector then the output will
            be of the shape `(len(x), 4, 1)`.
        """
        xi = jnp.asarray(xi)
        if xi.ndim != 3:
            raise ValueError(
                f"`xi` should be of size (npoints, 1, ndim); found {xi.shape}"
            )
        result = jnp.array(
            [
                0.25 * (1 - xi[:, :, 0]) * (1 - xi[:, :, 1]),
                0.25 * (1 + xi[:, :, 0]) * (1 - xi[:, :, 1]),
                0.25 * (1 + xi[:, :, 0]) * (1 + xi[:, :, 1]),
                0.25 * (1 - xi[:, :, 0]) * (1 + xi[:, :, 1]),
            ]
        )
        result = result.transpose(1, 0, 2)[..., jnp.newaxis]
        return result

    @classmethod
    def _shapefn(cls, xi: ArrayLike):
        """Evaluate linear shape function.

        Parameters
        ----------
        xi : float, array_like
            Locations of particles in natural coordinates to evaluate
            the function at. Expected shape is (npoints, 1, ndim)

        Returns
        -------
        array_like
            Evaluated shape function values. The shape of the returned
            array will depend on the input shape. For example, in the linear
            case, if the input is a scalar, the returned array will be of
            the shape `(1, 4, 1)` but if the input is a vector then the output will
            be of the shape `(len(x), 4, 1)`.
        """
        xi = jnp.asarray(xi)
        if xi.ndim != 3:
            raise ValueError(
                f"`xi` should be of size (npoints, 1, ndim); found {xi.shape}"
            )
        result = jnp.array(
            [
                0.25 * (1 - xi[:, :, 0]) * (1 - xi[:, :, 1]),
                0.25 * (1 + xi[:, :, 0]) * (1 - xi[:, :, 1]),
                0.25 * (1 + xi[:, :, 0]) * (1 + xi[:, :, 1]),
                0.25 * (1 - xi[:, :, 0]) * (1 + xi[:, :, 1]),
            ]
        )
        result = result.transpose(1, 0, 2)[..., jnp.newaxis]
        return result

    @classmethod
    def _shapefn_natural_grad(cls, xi: ArrayLike):
        """Calculate the gradient of shape function.

        This calculation is done in the natural coordinates.

        Parameters
        ----------
        x : float, array_like
            Locations of particles in natural coordinates to evaluate
            the function at.

        Returns
        -------
        array_like
            Evaluated gradient values of the shape function. The shape of
            the returned array will depend on the input shape. For example,
            in the linear case, if the input is a scalar, the returned array
            will be of the shape `(4, 2)`.
        """
        # result = vmap(jacobian(self.shapefn))(xi[..., jnp.newaxis]).squeeze()
        xi = jnp.asarray(xi)
        xi = xi.squeeze()
        result = jnp.array(
            [
                [-0.25 * (1 - xi[1]), -0.25 * (1 - xi[0])],
                [0.25 * (1 - xi[1]), -0.25 * (1 + xi[0])],
                [0.25 * (1 + xi[1]), 0.25 * (1 + xi[0])],
                [-0.25 * (1 + xi[1]), 0.25 * (1 - xi[0])],
            ],
        )
        return result

    def shapefn_grad(self, xi: ArrayLike, coords: ArrayLike):
        """Gradient of shape function in physical coordinates.

        Parameters
        ----------
        xi : float, array_like
            Locations of particles to evaluate in natural coordinates.
            Expected shape `(npoints, 1, ndim)`.
        coords : array_like
            Nodal coordinates to transform by. Expected shape
            `(npoints, 1, ndim)`

        Returns
        -------
        array_like
            Gradient of the shape function in physical coordinates at `xi`
        """
        xi = jnp.asarray(xi)
        coords = jnp.asarray(coords)
        if xi.ndim != 3:
            raise ValueError(
                f"`x` should be of size (npoints, 1, ndim); found {xi.shape}"
            )
        grad_sf = self._shapefn_natural_grad(xi)
        _jacobian = grad_sf.T @ coords.squeeze()

        result = grad_sf @ jnp.linalg.inv(_jacobian).T
        return result

    @classmethod
    def _shapefn_grad(cls, xi: ArrayLike, coords: ArrayLike):
        """Gradient of shape function in physical coordinates.

        Parameters
        ----------
        xi : float, array_like
            Locations of particles to evaluate in natural coordinates.
            Expected shape `(npoints, 1, ndim)`.
        coords : array_like
            Nodal coordinates to transform by. Expected shape
            `(npoints, 1, ndim)`

        Returns
        -------
        array_like
            Gradient of the shape function in physical coordinates at `xi`
        """
        xi = jnp.asarray(xi)
        coords = jnp.asarray(coords)
        if xi.ndim != 3:
            raise ValueError(
                f"`x` should be of size (npoints, 1, ndim); found {xi.shape}"
            )
        grad_sf = cls._shapefn_natural_grad(xi)
        _jacobian = grad_sf.T @ coords.squeeze()

        result = grad_sf @ jnp.linalg.inv(_jacobian).T
        return result

    @classmethod
    def _get_particles_element_ids(cls, particles, elements):
        """Set the element IDs for the particles.

        If the particle doesn't lie between the boundaries of any
        element, it sets the element index to -1.
        """

        def f(x, *, loc, nelements):
            xidl = (loc[:, :, 0] <= x[0, 0]).nonzero(size=loc.shape[0], fill_value=-1)[
                0
            ]
            yidl = (loc[:, :, 1] <= x[0, 1]).nonzero(size=loc.shape[0], fill_value=-1)[
                0
            ]
            lower_left = jnp.where(jnp.isin(xidl, yidl), xidl, -1).max()
            element_id = lower_left - lower_left // (nelements + 1)
            return element_id

        pf = partial(f, loc=elements.nodes.loc, nelements=elements.nelements[0])
        ids = vmap(pf)(particles.loc)
        return ids

    def set_particle_element_ids(
        self, elements: _ElementsState, particles: _ParticlesState
    ):
        """Set the element IDs for the particles.

        If the particle doesn't lie between the boundaries of any
        element, it sets the element index to -1.
        """

        @jit
        def f(x, *, loc, nelements):
            xidl = (loc[:, :, 0] <= x[0, 0]).nonzero(size=loc.shape[0], fill_value=-1)[
                0
            ]
            yidl = (loc[:, :, 1] <= x[0, 1]).nonzero(size=loc.shape[0], fill_value=-1)[
                0
            ]
            lower_left = jnp.where(jnp.isin(xidl, yidl), xidl, -1).max()
            element_id = lower_left - lower_left // (nelements + 1)
            return element_id

        pf = partial(f, loc=elements.nodes.loc, nelements=elements.nelements[0])
        ids = vmap(pf)(particles.loc)
        return particles.replace(element_ids=ids)

    def compute_internal_force(
        self, elements: _ElementState, particles: _ParticlesState
    ):
        r"""Update the nodal internal force based on particle mass.

        The nodal force is updated as a sum of internal forces for
        all particles mapped to the node.

        \[
            (f_{int})_i = -\sum_p V_p \sigma_p \nabla N_i(x_p)
        \]

        where \(\sigma_p\) is the stress at particle \(p\).

        Parameters
        ----------
        particles: diffmpm.particle._ParticlesState
            Particles to map to the nodal values.
        """

        @jit
        def _step(pid, args):
            (
                f_int,
                pvol,
                mapped_grads,
                el_nodes,
                pstress,
            ) = args
            force = jnp.zeros((mapped_grads.shape[1], 1, 2))
            force = force.at[:, 0, 0].set(
                mapped_grads[pid][:, 0] * pstress[pid][0]
                + mapped_grads[pid][:, 1] * pstress[pid][3]
            )
            force = force.at[:, 0, 1].set(
                mapped_grads[pid][:, 1] * pstress[pid][1]
                + mapped_grads[pid][:, 0] * pstress[pid][3]
            )
            update = -pvol[pid] * force
            f_int = f_int.at[el_nodes[pid]].add(update)
            return (
                f_int,
                pvol,
                mapped_grads,
                el_nodes,
                pstress,
            )

        # f_int = self.nodes.f_int.at[:].set(0)
        f_int = elements.nodes.f_int
        mapped_nodes = vmap(Partial(self.id_to_node_ids, elements.nelements[0]))(
            particles.element_ids
        ).squeeze(-1)
        mapped_coords = vmap(Partial(self.id_to_node_loc, elements))(
            particles.element_ids
        ).squeeze(2)
        mapped_grads = vmap(jit(self.shapefn_grad))(
            particles.reference_loc[:, jnp.newaxis, ...],
            mapped_coords,
        )
        args = (
            f_int,
            particles.volume,
            mapped_grads,
            mapped_nodes,
            particles.stress,
        )
        f_int, _, _, _, _ = lax.fori_loop(0, particles.nparticles, _step, args)
        return f_int, "f_int"

    @classmethod
    def _compute_internal_force(
        cls, nf_int, nloc, mapped_node_ids, pxi, pvol, pstress, nparticles
    ):
        r"""Update the nodal internal force based on particle mass.

        The nodal force is updated as a sum of internal forces for
        all particles mapped to the node.

        \[
            (f_{int})_i = -\sum_p V_p \sigma_p \nabla N_i(x_p)
        \]

        where \(\sigma_p\) is the stress at particle \(p\).

        Parameters
        ----------
        particles: diffmpm.particle._ParticlesState
            Particles to map to the nodal values.
        """

        @jit
        def _step(pid, args):
            (
                f_int,
                pvol,
                mapped_grads,
                el_nodes,
                pstress,
            ) = args
            force = jnp.zeros((mapped_grads.shape[1], 1, 2))
            force = force.at[:, 0, 0].set(
                mapped_grads[pid][:, 0] * pstress[pid][0]
                + mapped_grads[pid][:, 1] * pstress[pid][3]
            )
            force = force.at[:, 0, 1].set(
                mapped_grads[pid][:, 1] * pstress[pid][1]
                + mapped_grads[pid][:, 0] * pstress[pid][3]
            )
            update = -pvol[pid] * force
            f_int = f_int.at[el_nodes[pid]].add(update)
            return (
                f_int,
                pvol,
                mapped_grads,
                el_nodes,
                pstress,
            )

        # f_int = self.nodes.f_int.at[:].set(0)
        # f_int = elements.nodes.f_int
        mapped_nodes = mapped_node_ids.squeeze(-1)
        mapped_coords = nloc[mapped_nodes].squeeze(2)
        mapped_grads = vmap(jit(cls._shapefn_grad))(
            pxi[:, jnp.newaxis, ...],
            mapped_coords,
        )
        args = (
            nf_int,
            pvol,
            mapped_grads,
            mapped_nodes,
            pstress,
        )
        f_int, _, _, _, _ = lax.fori_loop(0, nparticles, _step, args)
        return f_int

    # Mapping from particles to nodes (P2G)
    @classmethod
    def _compute_nodal_mass(cls, mass, pmass, pxi, peids, mapped_node_ids, nparticles):
        r"""Compute the nodal mass based on particle mass.

        The nodal mass is updated as a sum of particle mass for
        all particles mapped to the node.

        \[
            (m)_i = \sum_p N_i(x_p) m_p
        \]

        Parameters
        ----------
        particles: diffmpm.particle.Particles
            Particles to map to the nodal values.
        """

        @jit
        def _step(pid, args):
            pmass, mass, mapped_pos, el_nodes = args
            mass = mass.at[el_nodes[pid]].add(pmass[pid] * mapped_pos[pid])
            return pmass, mass, mapped_pos, el_nodes

        mapped_positions = cls._shapefn(pxi)
        mapped_nodes = mapped_node_ids.squeeze(-1)
        args = (
            pmass,
            mass,
            mapped_positions,
            mapped_nodes,
        )
        _, mass, _, _ = lax.fori_loop(0, nparticles, _step, args)
        return mass

    def compute_nodal_mass(self, elements, particles: _ParticlesState):
        r"""Compute the nodal mass based on particle mass.

        The nodal mass is updated as a sum of particle mass for
        all particles mapped to the node.

        \[
            (m)_i = \sum_p N_i(x_p) m_p
        \]

        Parameters
        ----------
        particles: diffmpm.particle.Particles
            Particles to map to the nodal values.
        """

        @jit
        def _step(pid, args):
            pmass, mass, mapped_pos, el_nodes = args
            mass = mass.at[el_nodes[pid]].add(pmass[pid] * mapped_pos[pid])
            return pmass, mass, mapped_pos, el_nodes

        # mass = self.nodes.mass.at[:].set(0)
        mass = elements.nodes.mass
        mapped_positions = self.shapefn(particles.reference_loc)
        mapped_nodes = vmap(Partial(self.id_to_node_ids, elements.nelements[0]))(
            particles.element_ids
        ).squeeze(-1)
        args = (
            particles.mass,
            mass,
            mapped_positions,
            mapped_nodes,
        )
        _, mass, _, _ = lax.fori_loop(0, particles.nparticles, _step, args)
        # TODO: Return state instead of setting
        return mass, "mass"

    @classmethod
    def _compute_nodal_momentum(
        cls, nmom, pmass, pvel, pxi, peids, mapped_node_ids, nparticles
    ):
        r"""Compute the nodal mass based on particle mass.

        The nodal mass is updated as a sum of particle mass for
        all particles mapped to the node.

        \[
            (mv)_i = \sum_p N_i(x_p) (mv)_p
        \]

        Parameters
        ----------
        particles: diffmpm.particle.Particles
            Particles to map to the nodal values.
        """

        @jit
        def _step(pid, args):
            pmom, mom, mapped_pos, el_nodes = args
            new_mom = mom.at[el_nodes[pid]].add(mapped_pos[pid] @ pmom[pid])
            return pmom, new_mom, mapped_pos, el_nodes

        # curr_mom = elements.nodes.momentum.at[:].set(0)
        # curr_mom = elements.nodes.momentum
        mapped_nodes = mapped_node_ids.squeeze(-1)
        mapped_positions = cls._shapefn(pxi)
        args = (
            pmass * pvel,
            nmom,
            mapped_positions,
            mapped_nodes,
        )
        _, new_momentum, _, _ = lax.fori_loop(0, nparticles, _step, args)
        new_momentum = jnp.where(jnp.abs(new_momentum) < 1e-12, 0, new_momentum)
        return new_momentum

    def compute_nodal_momentum(self, elements, particles: _ParticlesState):
        r"""Compute the nodal mass based on particle mass.

        The nodal mass is updated as a sum of particle mass for
        all particles mapped to the node.

        \[
            (mv)_i = \sum_p N_i(x_p) (mv)_p
        \]

        Parameters
        ----------
        particles: diffmpm.particle.Particles
            Particles to map to the nodal values.
        """

        @jit
        def _step(pid, args):
            pmom, mom, mapped_pos, el_nodes = args
            new_mom = mom.at[el_nodes[pid]].add(mapped_pos[pid] @ pmom[pid])
            return pmom, new_mom, mapped_pos, el_nodes

        # curr_mom = elements.nodes.momentum.at[:].set(0)
        curr_mom = elements.nodes.momentum
        mapped_positions = self.shapefn(particles.reference_loc)
        mapped_nodes = vmap(Partial(self.id_to_node_ids, elements.nelements[0]))(
            particles.element_ids
        ).squeeze(-1)
        args = (
            particles.mass * particles.velocity,
            curr_mom,
            mapped_positions,
            mapped_nodes,
        )
        _, new_momentum, _, _ = lax.fori_loop(0, particles.nparticles, _step, args)
        new_momentum = jnp.where(jnp.abs(new_momentum) < 1e-12, 0, new_momentum)
        # TODO: Return state instead of setting
        return new_momentum, "momentum"

    @classmethod
    def _compute_nodal_velocity(cls, nmass, nmom, nvel):
        """Compute velocity using momentum."""
        velocity = jnp.where(
            nmass == 0,
            nvel,
            nmom / nmass,
        )
        velocity = jnp.where(
            jnp.abs(velocity) < 1e-12,
            0,
            velocity,
        )
        # TODO: Return state instead of setting
        return velocity

    def compute_velocity(self, elements, particles: _ParticlesState):
        """Compute velocity using momentum."""
        velocity = jnp.where(
            elements.nodes.mass == 0,
            elements.nodes.velocity,
            elements.nodes.momentum / elements.nodes.mass,
        )
        velocity = jnp.where(
            jnp.abs(velocity) < 1e-12,
            0,
            velocity,
        )
        # TODO: Return state instead of setting
        return velocity, "velocity"

    def compute_external_force(self, elements, particles: _ParticlesState):
        r"""Update the nodal external force based on particle f_ext.

        The nodal force is updated as a sum of particle external
        force for all particles mapped to the node.

        \[
            f_{ext})_i = \sum_p N_i(x_p) f_{ext}
        \]

        Parameters
        ----------
        particles: diffmpm.particle.Particles
            Particles to map to the nodal values.
        """

        @jit
        def _step(pid, args):
            f_ext, pf_ext, mapped_pos, el_nodes = args
            f_ext = f_ext.at[el_nodes[pid]].add(mapped_pos[pid] @ pf_ext[pid])
            return f_ext, pf_ext, mapped_pos, el_nodes

        # f_ext = elements.nodes.f_ext.at[:].set(0)
        f_ext = elements.nodes.f_ext
        mapped_positions = self.shapefn(particles.reference_loc)
        mapped_nodes = vmap(Partial(self.id_to_node_ids, elements.nelements[0]))(
            particles.element_ids
        ).squeeze(-1)
        args = (
            f_ext,
            particles.f_ext,
            mapped_positions,
            mapped_nodes,
        )
        f_ext, _, _, _ = lax.fori_loop(0, particles.nparticles, _step, args)
        # TODO: Return state instead of setting
        return f_ext, "f_ext"

    @classmethod
    def _compute_external_force(cls, f_ext, pf_ext, pxi, nparticles, mapped_node_ids):
        r"""Update the nodal external force based on particle f_ext.

        The nodal force is updated as a sum of particle external
        force for all particles mapped to the node.

        \[
            f_{ext})_i = \sum_p N_i(x_p) f_{ext}
        \]

        Parameters
        ----------
        particles: diffmpm.particle.Particles
            Particles to map to the nodal values.
        """

        @jit
        def _step(pid, args):
            f_ext, pf_ext, mapped_pos, el_nodes = args
            f_ext = f_ext.at[el_nodes[pid]].add(mapped_pos[pid] @ pf_ext[pid])
            return f_ext, pf_ext, mapped_pos, el_nodes

        # f_ext = elements.nodes.f_ext.at[:].set(0)
        mapped_positions = cls._shapefn(pxi)
        mapped_nodes = mapped_node_ids.squeeze(-1)
        args = (
            f_ext,
            pf_ext,
            mapped_positions,
            mapped_nodes,
        )
        f_ext, _, _, _ = lax.fori_loop(0, nparticles, _step, args)
        return f_ext

    def compute_body_force(
        self, elements, particles: _ParticlesState, gravity: ArrayLike
    ):
        r"""Update the nodal external force based on particle mass.

        The nodal force is updated as a sum of particle body
        force for all particles mapped to th

        \[
            (f_{ext})_i = (f_{ext})_i + \sum_p N_i(x_p) m_p g
        \]

        Parameters
        ----------
        particles: diffmpm.particle._ParticlesState
            Particles to map to the nodal values.
        """

        @jit
        def _step(pid, args):
            f_ext, pmass, mapped_pos, el_nodes, gravity = args
            f_ext = f_ext.at[el_nodes[pid]].add(
                mapped_pos[pid] @ (pmass[pid] * gravity)
            )
            return f_ext, pmass, mapped_pos, el_nodes, gravity

        mapped_positions = self.shapefn(particles.reference_loc)
        mapped_nodes = vmap(Partial(self.id_to_node_ids, elements.nelements[0]))(
            particles.element_ids
        ).squeeze(-1)
        args = (
            elements.nodes.f_ext,
            particles.mass,
            mapped_positions,
            mapped_nodes,
            gravity,
        )
        f_ext, _, _, _, _ = lax.fori_loop(0, particles.nparticles, _step, args)
        # TODO: Return state instead of setting
        return f_ext, "f_ext"

    @classmethod
    def _compute_body_force(
        cls, nf_ext, pmass, pxi, mapped_node_ids, nparticles, gravity: ArrayLike
    ):
        r"""Update the nodal external force based on particle mass.

        The nodal force is updated as a sum of particle body
        force for all particles mapped to th

        \[
            (f_{ext})_i = (f_{ext})_i + \sum_p N_i(x_p) m_p g
        \]

        Parameters
        ----------
        particles: diffmpm.particle._ParticlesState
            Particles to map to the nodal values.
        """

        @jit
        def _step(pid, args):
            f_ext, pmass, mapped_pos, el_nodes, gravity = args
            f_ext = f_ext.at[el_nodes[pid]].add(
                mapped_pos[pid] @ (pmass[pid] * gravity)
            )
            return f_ext, pmass, mapped_pos, el_nodes, gravity

        mapped_positions = cls._shapefn(pxi)
        mapped_nodes = mapped_node_ids.squeeze(-1)
        args = (
            nf_ext,
            pmass,
            mapped_positions,
            mapped_nodes,
            gravity,
        )
        f_ext, _, _, _, _ = lax.fori_loop(0, nparticles, _step, args)
        return f_ext

    def apply_concentrated_nodal_forces(
        self, elements, particles: _ParticlesState, curr_time: float
    ):
        """Apply concentrated nodal forces.

        Parameters
        ----------
        particles: _ParticlesState
            Particles in the simulation.
        curr_time: float
            Current time in the simulation.
        """

        def _func(cnf, *, f_ext):
            factor = cnf.function.value(curr_time)
            f_ext = f_ext.at[cnf.node_ids, 0, cnf.dir].add(factor * cnf.force)
            return f_ext

        if elements.concentrated_nodal_forces:
            partial_func = partial(_func, f_ext=elements.nodes.f_ext)
            _out = tree_map(
                partial_func,
                elements.concentrated_nodal_forces,
                is_leaf=lambda x: isinstance(x, NodalForce),
            )

            def _f(x, *, orig):
                return jnp.where(x == orig, 0, x)

            # This assumes that the nodal forces are not overlapping, i.e.
            # no node will be acted by 2 forces in the same direction.
            _step_1 = tree_map(partial(_f, orig=elements.nodes.f_ext), _out)
            _step_2 = tree_reduce(lambda x, y: x + y, _step_1)
            f_ext = jnp.where(_step_2 == 0, elements.nodes.f_ext, _step_2)
            # TODO: Return state instead of setting
            return f_ext, "f_ext"

    @classmethod
    def _apply_concentrated_nodal_forces(
        self, nf_ext, concentrated_forces, curr_time: float
    ):
        """Apply concentrated nodal forces.

        Parameters
        ----------
        particles: _ParticlesState
            Particles in the simulation.
        curr_time: float
            Current time in the simulation.
        """

        def _func(cnf, f_ext):
            factor = cnf.function.value(curr_time)
            f_ext = f_ext.at[cnf.node_ids, 0, cnf.dir].add(factor * cnf.force)
            return f_ext

        _out = tree_map(
            _func,
            concentrated_forces,
            [nf_ext] * len(concentrated_forces),
            is_leaf=lambda x: isinstance(x, NodalForce) or isinstance(x, Array),
        )

        def _f(x, *, orig):
            return jnp.where(x == orig, 0, x)

        # This assumes that the nodal forces are not overlapping, i.e.
        # no node will be acted by 2 forces in the same direction.
        _step_1 = tree_map(partial(_f, orig=nf_ext), _out)
        _step_2 = tree_reduce(lambda x, y: x + y, _step_1)
        f_ext = jnp.where(_step_2 == 0, nf_ext, _step_2)
        return f_ext

    @classmethod
    def _apply_particle_traction_forces(
        cls, pxi, mapped_node_ids, nf_ext, ptraction, nparticles
    ):
        """Apply concentrated nodal forces.

        Parameters
        ----------
        particles: Particles
            Particles in the simulation.
        """

        @jit
        def _step(pid, args):
            f_ext, ptraction, mapped_pos, el_nodes = args
            f_ext = f_ext.at[el_nodes[pid]].add(mapped_pos[pid] @ ptraction[pid])
            return f_ext, ptraction, mapped_pos, el_nodes

        mapped_positions = cls._shapefn(pxi)
        mapped_nodes = mapped_node_ids.squeeze(-1)
        args = (
            nf_ext,
            ptraction,
            mapped_positions,
            mapped_nodes,
        )
        f_ext, _, _, _ = lax.fori_loop(0, nparticles, _step, args)
        return f_ext

    def apply_particle_traction_forces(self, elements, particles: _ParticlesState):
        """Apply concentrated nodal forces.

        Parameters
        ----------
        particles: Particles
            Particles in the simulation.
        """

        @jit
        def _step(pid, args):
            f_ext, ptraction, mapped_pos, el_nodes = args
            f_ext = f_ext.at[el_nodes[pid]].add(mapped_pos[pid] @ ptraction[pid])
            return f_ext, ptraction, mapped_pos, el_nodes

        mapped_positions = self.shapefn(particles.reference_loc)
        mapped_nodes = vmap(Partial(self.id_to_node_ids, elements.nelements[0]))(
            particles.element_ids
        ).squeeze(-1)
        args = (
            elements.nodes.f_ext,
            particles.traction,
            mapped_positions,
            mapped_nodes,
        )
        f_ext, _, _, _ = lax.fori_loop(0, particles.nparticles, _step, args)
        # TODO: Return state instead of setting
        return f_ext, "f_ext"

    def update_nodal_acceleration(
        self, elements, particles: _ParticlesState, dt: float, *args
    ):
        """Update the nodal momentum based on total force on nodes."""
        total_force = (
            elements.nodes.f_int + elements.nodes.f_ext + elements.nodes.f_damp
        )
        acceleration = elements.nodes.acceleration.at[:].set(
            jnp.nan_to_num(jnp.divide(total_force, elements.nodes.mass))
        )
        if elements.constraints:
            acceleration = self._apply_boundary_constraints_acc(elements, acceleration)
        acceleration = jnp.where(
            jnp.abs(acceleration) < 1e-12,
            0,
            acceleration,
        )
        return acceleration, "acceleration"

    @classmethod
    def _update_nodal_acceleration(
        cls,
        total_force,
        nacc,
        nmass,
        constraints,
        tol,
    ):
        """Update the nodal momentum based on total force on nodes."""
        acceleration = jnp.nan_to_num(jnp.divide(total_force, nmass))
        if constraints:
            acceleration = cls._apply_boundary_constraints_acc(
                constraints, acceleration
            )
        acceleration = jnp.where(
            jnp.abs(acceleration) < tol,
            0,
            acceleration,
        )
        return acceleration

    def update_nodal_velocity(
        self, elements, particles: _ParticlesState, dt: float, *args
    ):
        """Update the nodal momentum based on total force on nodes."""
        total_force = (
            elements.nodes.f_int + elements.nodes.f_ext + elements.nodes.f_damp
        )
        acceleration = jnp.nan_to_num(jnp.divide(total_force, elements.nodes.mass))

        velocity = elements.nodes.velocity + acceleration * dt
        if elements.constraints:
            velocity = self._apply_boundary_constraints_vel(elements, velocity)
        velocity = jnp.where(
            jnp.abs(velocity) < 1e-12,
            0,
            velocity,
        )
        return velocity, "velocity"

    @classmethod
    def _update_nodal_velocity(cls, total_force, nvel, nmass, constraints, dt, tol):
        """Update the nodal momentum based on total force on nodes."""
        acceleration = jnp.nan_to_num(jnp.divide(total_force, nmass))

        velocity = nvel + acceleration * dt
        if constraints:
            velocity = cls._apply_boundary_constraints_vel(constraints, velocity)
        velocity = jnp.where(
            jnp.abs(velocity) < tol,
            0,
            velocity,
        )
        return velocity

    def update_nodal_momentum(
        self, elements, particles: _ParticlesState, dt: float, *args
    ):
        """Update the nodal momentum based on total force on nodes."""
        momentum = elements.nodes.momentum.at[:].set(
            elements.nodes.mass * elements.nodes.velocity
        )
        momentum = jnp.where(
            jnp.abs(momentum) < 1e-12,
            0,
            momentum,
        )
        return momentum, "momentum"

    @classmethod
    def _update_nodal_momentum(cls, nmass, nvel, constraints, tol):
        """Update the nodal momentum based on total force on nodes."""
        momentum = nmass * nvel
        momentum = jnp.where(
            jnp.abs(momentum) < tol,
            0,
            momentum,
        )
        return momentum

    @classmethod
    def _apply_boundary_constraints_vel(cls, constraints, vel, *args):
        """Apply boundary conditions for nodal velocity."""

        # This assumes that the constraints don't have overlapping
        # conditions. In case it does, only the first constraint will
        # be applied.
        def _func2(constraint, *, orig):
            return constraint[1].apply_vel(orig, constraint[0])

        partial_func = partial(_func2, orig=vel)
        _out = tree_map(
            partial_func, constraints, is_leaf=lambda x: isinstance(x, tuple)
        )

        def _f(x, *, orig):
            return jnp.where(x == orig, jnp.nan, x)

        _pf = partial(_f, orig=vel)
        _step_1 = tree_map(_pf, _out)
        vel = tree_reduce(
            lambda x, y: jnp.where(jnp.isnan(y), x, y),
            [vel, _step_1],
        )
        return vel

    @classmethod
    def _apply_boundary_constraints_mom(cls, constraints, mom, mass, *args):
        """Apply boundary conditions for nodal momentum."""

        # This assumes that the constraints don't have overlapping
        # conditions. In case it does, only the first constraint will
        # be applied.
        def _func2(constraint, *, mom, mass):
            return constraint[1].apply_mom(mom, mass, constraint[0])

        partial_func = partial(_func2, mom=mom, mass=mass)
        _out = tree_map(
            partial_func, constraints, is_leaf=lambda x: isinstance(x, tuple)
        )

        def _f(x, *, orig):
            return jnp.where(x == orig, jnp.nan, x)

        _pf = partial(_f, orig=mom)
        _step_1 = tree_map(_pf, _out)
        mom = tree_reduce(
            lambda x, y: jnp.where(jnp.isnan(y), x, y),
            [mom, _step_1],
        )
        return mom

    @classmethod
    def _apply_boundary_constraints_acc(cls, constraints, orig, *args):
        """Apply boundary conditions for nodal acceleration."""

        # This assumes that the constraints don't have overlapping
        # conditions. In case it does, only the first constraint will
        # be applied.
        def _func2(constraint, *, orig):
            return constraint[1].apply_acc(orig, constraint[0])

        partial_func = partial(_func2, orig=orig)
        _out = tree_map(
            partial_func, constraints, is_leaf=lambda x: isinstance(x, tuple)
        )

        def _f(x, *, orig):
            return jnp.where(x == orig, jnp.nan, x)

        _pf = partial(_f, orig=orig)
        _step_1 = tree_map(_pf, _out)
        acc = tree_reduce(
            lambda x, y: jnp.where(jnp.isnan(y), x, y),
            [orig, _step_1],
        )
        return acc

    @classmethod
    def _apply_boundary_constraints(cls, nvel, nmom, nacc, nmass, constraints, *args):
        if constraints:
            vel = cls._apply_boundary_constraints_vel(constraints, nvel, *args)
            mom = cls._apply_boundary_constraints_mom(constraints, nmom, nmass, *args)
            acc = cls._apply_boundary_constraints_acc(constraints, nacc, *args)
            return vel, mom, acc

    def apply_boundary_constraints(self, elements, *args):
        if elements.constraints:
            vel = self._apply_boundary_constraints_vel(
                elements, elements.nodes.velocity, *args
            )
            mom = self._apply_boundary_constraints_mom(
                elements, elements.nodes.momentum, elements.nodes.mass, *args
            )
            acc = self._apply_boundary_constraints_acc(
                elements, elements.nodes.acceleration, *args
            )

            return elements.nodes.replace(velocity=vel, momentum=mom, acceleration=acc)

    @classmethod
    def _compute_volume(cls, el_len, evol):
        """Compute volume of all elements."""
        a = c = el_len[1]
        b = d = el_len[0]
        p = q = jnp.sqrt(a**2 + b**2)
        vol = 0.25 * jnp.sqrt(4 * p * p * q * q - (a * a + c * c - b * b - d * d) ** 2)
        volume = jnp.ones_like(evol) * vol
        return volume

    def compute_volume(self, elements, *args):
        """Compute volume of all elements."""
        a = c = elements.el_len[1]
        b = d = elements.el_len[0]
        p = q = jnp.sqrt(a**2 + b**2)
        vol = 0.25 * jnp.sqrt(4 * p * p * q * q - (a * a + c * c - b * b - d * d) ** 2)
        volume = jnp.ones_like(elements.volume) * vol
        return elements.replace(volume=volume)
