from __future__ import annotations

import abc
import itertools
from functools import partial
from typing import TYPE_CHECKING, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from diffmpm.particle import Particles

import jax.numpy as jnp
from jax import Array, jacobian, jit, lax, vmap
from jax.tree_util import register_pytree_node_class, tree_map
from jax.typing import ArrayLike

from diffmpm.constraint import Constraint
from diffmpm.forces import NodalForce
from diffmpm.node import Nodes

__all__ = ["_Element", "Linear1D", "Quadrilateral4Node"]


class _Element(abc.ABC):
    """Base element class that is inherited by all types of Elements."""

    nodes: Nodes
    total_elements: int
    concentrated_nodal_forces: Sequence
    volume: Array

    @abc.abstractmethod
    def id_to_node_ids(self, id: ArrayLike) -> Array:
        """Node IDs corresponding to element `id`.

        This method is implemented by each of the subclass.

        Parameters
        ----------
        id : int
            Element ID.

        Returns
        -------
        ArrayLike
            Nodal IDs of the element.
        """
        ...

    @jit
    def id_to_node_loc(self, id: ArrayLike) -> Array:
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
        node_ids = self.id_to_node_ids(id).squeeze()
        return self.nodes.loc[node_ids]

    @jit
    def id_to_node_vel(self, id: ArrayLike) -> Array:
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
        node_ids = self.id_to_node_ids(id).squeeze()
        return self.nodes.velocity[node_ids]

    def tree_flatten(self):
        children = (self.nodes, self.volume)
        aux_data = (
            self.nelements,
            self.total_elements,
            self.el_len,
            self.constraints,
            self.concentrated_nodal_forces,
            self.initialized,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            aux_data[0],
            aux_data[1],
            aux_data[2],
            aux_data[3],
            nodes=children[0],
            concentrated_nodal_forces=aux_data[4],
            initialized=aux_data[5],
            volume=children[1],
        )

    @abc.abstractmethod
    def shapefn(self, xi: ArrayLike):
        """Evaluate Shape function for element type."""
        ...

    @abc.abstractmethod
    def shapefn_grad(self, xi: ArrayLike, coords: ArrayLike):
        """Evaluate gradient of shape function for element type."""
        ...

    @abc.abstractmethod
    def set_particle_element_ids(self, particles: Particles):
        """Set the element IDs that particles are present in."""
        ...

    # Mapping from particles to nodes (P2G)
    def compute_nodal_mass(self, particles: Particles):
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

        self.nodes.mass = self.nodes.mass.at[:].set(0)
        mapped_positions = self.shapefn(particles.reference_loc)
        mapped_nodes = vmap(self.id_to_node_ids)(particles.element_ids).squeeze(-1)
        args = (
            particles.mass,
            self.nodes.mass,
            mapped_positions,
            mapped_nodes,
        )
        _, self.nodes.mass, _, _ = lax.fori_loop(0, len(particles), _step, args)

    def compute_nodal_momentum(self, particles: Particles):
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
            mom = mom.at[el_nodes[pid]].add(mapped_pos[pid] @ pmom[pid])
            return pmom, mom, mapped_pos, el_nodes

        self.nodes.momentum = self.nodes.momentum.at[:].set(0)
        mapped_positions = self.shapefn(particles.reference_loc)
        mapped_nodes = vmap(self.id_to_node_ids)(particles.element_ids).squeeze(-1)
        args = (
            particles.mass * particles.velocity,
            self.nodes.momentum,
            mapped_positions,
            mapped_nodes,
        )
        _, self.nodes.momentum, _, _ = lax.fori_loop(0, len(particles), _step, args)
        self.nodes.momentum = jnp.where(
            jnp.abs(self.nodes.momentum) < 1e-12,
            jnp.zeros_like(self.nodes.momentum),
            self.nodes.momentum,
        )

    def compute_velocity(self, particles: Particles):
        """Compute velocity using momentum."""
        self.nodes.velocity = jnp.where(
            self.nodes.mass == 0,
            self.nodes.velocity,
            self.nodes.momentum / self.nodes.mass,
        )
        self.nodes.velocity = jnp.where(
            jnp.abs(self.nodes.velocity) < 1e-12,
            jnp.zeros_like(self.nodes.velocity),
            self.nodes.velocity,
        )

    def compute_external_force(self, particles: Particles):
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

        self.nodes.f_ext = self.nodes.f_ext.at[:].set(0)
        mapped_positions = self.shapefn(particles.reference_loc)
        mapped_nodes = vmap(self.id_to_node_ids)(particles.element_ids).squeeze(-1)
        args = (
            self.nodes.f_ext,
            particles.f_ext,
            mapped_positions,
            mapped_nodes,
        )
        self.nodes.f_ext, _, _, _ = lax.fori_loop(0, len(particles), _step, args)

    def compute_body_force(self, particles: Particles, gravity: ArrayLike):
        r"""Update the nodal external force based on particle mass.

        The nodal force is updated as a sum of particle body
        force for all particles mapped to th

        \[
            (f_{ext})_i = (f_{ext})_i + \sum_p N_i(x_p) m_p g
        \]

        Parameters
        ----------
        particles: diffmpm.particle.Particles
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
        mapped_nodes = vmap(self.id_to_node_ids)(particles.element_ids).squeeze(-1)
        args = (
            self.nodes.f_ext,
            particles.mass,
            mapped_positions,
            mapped_nodes,
            gravity,
        )
        self.nodes.f_ext, _, _, _, _ = lax.fori_loop(0, len(particles), _step, args)

    def apply_concentrated_nodal_forces(self, particles: Particles, curr_time: float):
        """Apply concentrated nodal forces.

        Parameters
        ----------
        particles: Particles
            Particles in the simulation.
        curr_time: float
            Current time in the simulation.
        """

        def _func(cnf, *, nodes):
            factor = cnf.function.value(curr_time)
            nodes.f_ext = nodes.f_ext.at[cnf.node_ids, 0, cnf.dir].add(
                factor * cnf.force
            )

        partial_func = partial(_func, nodes=self.nodes)
        tree_map(
            partial_func,
            self.concentrated_nodal_forces,
            is_leaf=lambda x: isinstance(x, NodalForce),
        )

    def apply_particle_traction_forces(self, particles: Particles):
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
        mapped_nodes = vmap(self.id_to_node_ids)(particles.element_ids).squeeze(-1)
        args = (self.nodes.f_ext, particles.traction, mapped_positions, mapped_nodes)
        self.nodes.f_ext, _, _, _ = lax.fori_loop(0, len(particles), _step, args)

    def update_nodal_acceleration_velocity(
        self, particles: Particles, dt: float, *args
    ):
        """Update the nodal momentum based on total force on nodes."""
        total_force = self.nodes.get_total_force()
        self.nodes.acceleration = self.nodes.acceleration.at[:].set(
            jnp.nan_to_num(jnp.divide(total_force, self.nodes.mass))
        )
        self.nodes.velocity = self.nodes.velocity.at[:].add(
            self.nodes.acceleration * dt
        )
        self.apply_boundary_constraints()
        self.nodes.momentum = self.nodes.momentum.at[:].set(
            self.nodes.mass * self.nodes.velocity
        )
        self.nodes.velocity = jnp.where(
            jnp.abs(self.nodes.velocity) < 1e-12,
            jnp.zeros_like(self.nodes.velocity),
            self.nodes.velocity,
        )
        self.nodes.acceleration = jnp.where(
            jnp.abs(self.nodes.acceleration) < 1e-12,
            jnp.zeros_like(self.nodes.acceleration),
            self.nodes.acceleration,
        )

    def apply_boundary_constraints(self, *args):
        """Apply boundary conditions for nodal velocity."""
        for ids, constraint in self.constraints:
            constraint.apply(self.nodes, ids)

    def apply_force_boundary_constraints(self, *args):
        """Apply boundary conditions for nodal forces."""
        self.nodes.f_int = self.nodes.f_int.at[self.constraints[0][0]].set(0)
        self.nodes.f_ext = self.nodes.f_ext.at[self.constraints[0][0]].set(0)
        self.nodes.f_damp = self.nodes.f_damp.at[self.constraints[0][0]].set(0)


@register_pytree_node_class
class Linear1D(_Element):
    """Container for 1D line elements (and nodes).

        Element ID:            0     1     2     3
        Mesh:               +-----+-----+-----+-----+
        Node IDs:           0     1     2     3     4

    where

        + : Nodes
        +-----+ : An element

    """

    def __init__(
        self,
        nelements: int,
        total_elements: int,
        el_len: float,
        constraints: Sequence[Tuple[ArrayLike, Constraint]],
        nodes: Optional[Nodes] = None,
        concentrated_nodal_forces: Sequence = [],
        initialized: Optional[bool] = None,
        volume: Optional[ArrayLike] = None,
    ):
        """Initialize Linear1D.

        Parameters
        ----------
        nelements : int
            Number of elements.
        total_elements : int
            Total number of elements (same as `nelements` for 1D)
        el_len : float
            Length of each element.
        constraints: list
            A list of constraints where each element is a tuple of type
            `(node_ids, diffmpm.Constraint)`. Here, `node_ids` correspond to
            the node IDs where `diffmpm.Constraint` should be applied.
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
        self.nelements = nelements
        self.total_elements = nelements
        self.el_len = el_len
        if nodes is None:
            self.nodes = Nodes(
                nelements + 1,
                jnp.arange(nelements + 1).reshape(-1, 1, 1) * el_len,
            )
        else:
            self.nodes = nodes

        # self.boundary_nodes = boundary_nodes
        self.constraints = constraints
        self.concentrated_nodal_forces = concentrated_nodal_forces
        if initialized is None:
            self.volume = jnp.ones((self.total_elements, 1, 1))
        else:
            self.volume = jnp.asarray(volume)
        self.initialized = True

    def id_to_node_ids(self, id: ArrayLike):
        """Node IDs corresponding to element `id`.

        Parameters
        ----------
        id : int
            Element ID.

        Returns
        -------
        ArrayLike
            Nodal IDs of the element. Shape of returned
            array is `(2, 1)`
        """
        return jnp.array([id, id + 1]).reshape(2, 1)

    def shapefn(self, xi: ArrayLike):
        """Evaluate linear shape function.

        Parameters
        ----------
        xi : float, array_like
            Locations of particles in natural coordinates to evaluate
            the function at. Expected shape is `(npoints, 1, ndim)`

        Returns
        -------
        array_like
            Evaluated shape function values. The shape of the returned
            array will depend on the input shape. For example, in the linear
            case, if the input is a scalar, the returned array will be of
            the shape `(1, 2, 1)` but if the input is a vector then the output will
            be of the shape `(len(x), 2, 1)`.
        """
        xi = jnp.asarray(xi)
        if xi.ndim != 3:
            raise ValueError(
                f"`xi` should be of size (npoints, 1, ndim); found {xi.shape}"
            )
        result = jnp.array([0.5 * (1 - xi), 0.5 * (1 + xi)]).transpose(1, 0, 2, 3)
        return result

    def _shapefn_natural_grad(self, xi: ArrayLike):
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
            will be of the shape `(2, 1)`.
        """
        xi = jnp.asarray(xi)
        result = vmap(jacobian(self.shapefn))(xi[..., jnp.newaxis]).squeeze()

        # TODO: The following code tries to evaluate vmap even if
        # the predicate condition is true, not sure why.
        # result = lax.cond(
        #     jnp.isscalar(x),
        #     jacobian(self.shapefn),
        #     vmap(jacobian(self.shapefn)),
        #     xi
        # )
        return result.reshape(2, 1)

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
        _jacobian = grad_sf.T @ coords

        result = grad_sf @ jnp.linalg.inv(_jacobian).T
        return result

    def set_particle_element_ids(self, particles):
        """Set the element IDs for the particles.

        If the particle doesn't lie between the boundaries of any
        element, it sets the element index to -1.
        """

        @jit
        def f(x):
            idl = (
                len(self.nodes.loc)
                - 1
                - jnp.asarray(self.nodes.loc[::-1] <= x).nonzero(size=1, fill_value=-1)[
                    0
                ][-1]
            )
            idg = (
                jnp.asarray(self.nodes.loc > x).nonzero(size=1, fill_value=-1)[0][0] - 1
            )
            return (idl, idg)

        ids = vmap(f)(particles.loc)
        particles.element_ids = jnp.where(
            ids[0] == ids[1], ids[0], jnp.ones_like(ids[0]) * -1
        )

    def compute_volume(self, *args):
        """Compute volume of all elements."""
        vol = jnp.ediff1d(self.nodes.loc)
        self.volume = jnp.ones((self.total_elements, 1, 1)) * vol

    def compute_internal_force(self, particles):
        r"""Update the nodal internal force based on particle mass.

        The nodal force is updated as a sum of internal forces for
        all particles mapped to the node.

        \[
            (f_{int})_i = -\sum_p V_p \sigma_p \nabla N_i(x_p)
        \]

        where \(\sigma_p\) is the stress at particle \(p\).

        Parameters
        ----------
        particles: diffmpm.particle.Particles
            Particles to map to the nodal values.
        """

        def _step(pid, args):
            (
                f_int,
                pvol,
                mapped_grads,
                el_nodes,
                pstress,
            ) = args
            # TODO: correct matrix multiplication for n-d
            # update = -(pvol[pid]) * pstress[pid] @ mapped_grads[pid]
            update = -pvol[pid] * pstress[pid][0] * mapped_grads[pid]
            f_int = f_int.at[el_nodes[pid]].add(update[..., jnp.newaxis])
            return (
                f_int,
                pvol,
                mapped_grads,
                el_nodes,
                pstress,
            )

        self.nodes.f_int = self.nodes.f_int.at[:].set(0)
        mapped_nodes = vmap(self.id_to_node_ids)(particles.element_ids).squeeze(-1)
        mapped_coords = vmap(self.id_to_node_loc)(particles.element_ids).squeeze(2)
        mapped_grads = vmap(self.shapefn_grad)(
            particles.reference_loc[:, jnp.newaxis, ...],
            mapped_coords,
        )
        args = (
            self.nodes.f_int,
            particles.volume,
            mapped_grads,
            mapped_nodes,
            particles.stress,
        )
        self.nodes.f_int, _, _, _, _ = lax.fori_loop(0, len(particles), _step, args)


@register_pytree_node_class
class Quadrilateral4Node(_Element):
    r"""Container for 2D quadrilateral elements with 4 nodes.

    Nodes and elements are numbered as

                 15 +---+---+---+---+ 19
                    | 8 | 9 | 10| 11|
                 10 +---+---+---+---+ 14
                    | 4 | 5 | 6 | 7 |
                  5 +---+---+---+---+ 9
                    | 0 | 1 | 2 | 3 |
                    +---+---+---+---+
                    0   1   2   3   4

    where

            + : Nodes
            +---+
            |   | : An element
            +---+
    """

    def __init__(
        self,
        nelements: int,
        total_elements: int,
        el_len: float,
        constraints: Sequence[Tuple[ArrayLike, Constraint]],
        nodes: Optional[Nodes] = None,
        concentrated_nodal_forces: Sequence = [],
        initialized: Optional[bool] = None,
        volume: Optional[ArrayLike] = None,
    ) -> None:
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
        self.nelements = jnp.asarray(nelements)
        self.el_len = jnp.asarray(el_len)
        self.total_elements = total_elements

        if nodes is None:
            total_nodes = jnp.prod(self.nelements + 1)
            coords = jnp.asarray(
                list(
                    itertools.product(
                        jnp.arange(self.nelements[1] + 1),
                        jnp.arange(self.nelements[0] + 1),
                    )
                )
            )
            node_locations = (
                jnp.asarray([coords[:, 1], coords[:, 0]]).T * self.el_len
            ).reshape(-1, 1, 2)
            self.nodes = Nodes(int(total_nodes), node_locations)
        else:
            self.nodes = nodes

        self.constraints = constraints
        self.concentrated_nodal_forces = concentrated_nodal_forces
        if initialized is None:
            self.volume = jnp.ones((self.total_elements, 1, 1))
        else:
            self.volume = jnp.asarray(volume)
        self.initialized = True

    @jit
    def id_to_node_ids(self, id: ArrayLike):
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
        lower_left = (id // self.nelements[0]) * (
            self.nelements[0] + 1
        ) + id % self.nelements[0]
        result = jnp.asarray(
            [
                lower_left,
                lower_left + 1,
                lower_left + self.nelements[0] + 2,
                lower_left + self.nelements[0] + 1,
            ]
        )
        return result.reshape(4, 1)

    @jit
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

    @jit
    def _shapefn_natural_grad(self, xi: ArrayLike):
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

    @jit
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

    def set_particle_element_ids(self, particles: Particles):
        """Set the element IDs for the particles.

        If the particle doesn't lie between the boundaries of any
        element, it sets the element index to -1.
        """

        @jit
        def f(x):
            xidl = (self.nodes.loc[:, :, 0] <= x[0, 0]).nonzero(
                size=len(self.nodes.loc), fill_value=-1
            )[0]
            yidl = (self.nodes.loc[:, :, 1] <= x[0, 1]).nonzero(
                size=len(self.nodes.loc), fill_value=-1
            )[0]
            lower_left = jnp.where(jnp.isin(xidl, yidl), xidl, -1).max()
            element_id = lower_left - lower_left // (self.nelements[0] + 1)
            return element_id

        ids = vmap(f)(particles.loc)
        particles.element_ids = ids

    def compute_internal_force(self, particles: Particles):
        r"""Update the nodal internal force based on particle mass.

        The nodal force is updated as a sum of internal forces for
        all particles mapped to the node.

        \[
            (f_{int})_i = -\sum_p V_p \sigma_p \nabla N_i(x_p)
        \]

        where \(\sigma_p\) is the stress at particle \(p\).

        Parameters
        ----------
        particles: diffmpm.particle.Particles
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

        self.nodes.f_int = self.nodes.f_int.at[:].set(0)
        mapped_nodes = vmap(self.id_to_node_ids)(particles.element_ids).squeeze(-1)
        mapped_coords = vmap(self.id_to_node_loc)(particles.element_ids).squeeze(2)
        mapped_grads = vmap(self.shapefn_grad)(
            particles.reference_loc[:, jnp.newaxis, ...],
            mapped_coords,
        )
        args = (
            self.nodes.f_int,
            particles.volume,
            mapped_grads,
            mapped_nodes,
            particles.stress,
        )
        self.nodes.f_int, _, _, _, _ = lax.fori_loop(0, len(particles), _step, args)

    def compute_volume(self, *args):
        """Compute volume of all elements."""
        a = c = self.el_len[1]
        b = d = self.el_len[0]
        p = q = jnp.sqrt(a**2 + b**2)
        vol = 0.25 * jnp.sqrt(4 * p * p * q * q - (a * a + c * c - b * b - d * d) ** 2)
        self.volume = self.volume.at[:].set(vol)
