import abc
import itertools
from typing import Sequence, Tuple

import jax.numpy as jnp
from jax import jacobian, jit, lax, vmap
from jax.tree_util import register_pytree_node_class

from diffmpm.node import Nodes


class _Element(abc.ABC):
    @abc.abstractmethod
    def id_to_node_ids(self):
        ...

    @abc.abstractmethod
    def id_to_node_loc(self):
        ...

    @abc.abstractmethod
    def id_to_node_vel(self):
        ...

    def tree_flatten(self):
        children = (self.nodes,)
        aux_data = (self.nelements, self.el_len, self.boundary_nodes)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*aux_data, nodes=children[0])

    @abc.abstractmethod
    def shapefn(self):
        ...

    @abc.abstractmethod
    def shapefn_grad(self):
        ...

    # Mapping from particles to nodes (P2G)
    def compute_nodal_mass(self, particles):
        r"""
        Compute the nodal mass based on particle mass.

        The nodal mass is updated as a sum of particle mass for
        all particles mapped to the node.

        :math:`(m)_i = \sum_p N_i(x_p) m_p`

        Arguments
        ---------
        particles: diffmpm.particle.Particles
            Particles to map to the nodal values.
        """

        def _step(pid, args):
            pmass, mass, mapped_pos, el_nodes = args
            mass = mass.at[el_nodes[pid]].add(pmass[pid] * mapped_pos[pid])
            return pmass, mass, mapped_pos, el_nodes

        self.nodes.mass = self.nodes.mass.at[:].set(0)
        mapped_positions = self.shapefn(particles.reference_loc)
        mapped_nodes = vmap(self.id_to_node_ids)(particles.element_ids).squeeze(
            -1
        )
        args = (
            particles.mass,
            self.nodes.mass,
            mapped_positions,
            mapped_nodes,
        )
        _, self.nodes.mass, _, _ = lax.fori_loop(0, len(particles), _step, args)

    def compute_nodal_momentum(self, particles):
        r"""
        Compute the nodal mass based on particle mass.

        The nodal mass is updated as a sum of particle mass for
        all particles mapped to the node.

        :math:`(mv)_i = \sum_p N_i(x_p) (mv)_p`

        Arguments
        ---------
        particles: diffmpm.particle.Particles
            Particles to map to the nodal values.
        """

        def _step(pid, args):
            pmom, mom, mapped_pos, el_nodes = args
            mom = mom.at[el_nodes[pid]].add(mapped_pos[pid] @ pmom[pid])
            return pmom, mom, mapped_pos, el_nodes

        self.nodes.momentum = self.nodes.momentum.at[:].set(0)
        mapped_positions = self.shapefn(particles.reference_loc)
        mapped_nodes = vmap(self.id_to_node_ids)(particles.element_ids).squeeze(
            -1
        )
        args = (
            particles.mass * particles.velocity,
            self.nodes.momentum,
            mapped_positions,
            mapped_nodes,
        )
        _, self.nodes.momentum, _, _ = lax.fori_loop(
            0, len(particles), _step, args
        )

    def compute_nodal_velocity(self, particles):
        r"""
        Compute the nodal velocity based on particle velocity.

        The nodal mass is updated as a sum of particle mass for
        all particles mapped to the node.

        :math:`v_i = \sum_p N_i(x_p) v_p`

        Arguments
        ---------
        particles: diffmpm.particle.Particles
            Particles to map to the nodal values.
        """

        def _step(pid, args):
            pvel, pmass, vel, mass, mapped_pos, el_nodes = args
            vel = vel.at[el_nodes[pid]].add(
                jnp.divide(mapped_pos[pid], mass[el_nodes[pid]])
                @ pvel[pid]
                * pmass[pid]
            )
            return pvel, pmass, vel, mass, mapped_pos, el_nodes

        self.nodes.velocity = self.nodes.velocity.at[:].set(0)
        mapped_positions = self.shapefn(particles.reference_loc)
        mapped_nodes = vmap(self.id_to_node_ids)(particles.element_ids).squeeze(
            -1
        )
        args = (
            particles.velocity,
            particles.mass,
            self.nodes.velocity,
            self.nodes.mass,
            mapped_positions,
            mapped_nodes,
        )
        _, _, self.nodes.velocity, _, _, _ = lax.fori_loop(
            0, len(particles), _step, args
        )

    def compute_external_force(self, particles):
        r"""
        Update the nodal external force based on particle f_ext.

        The nodal force is updated as a sum of particle external
        force for all particles mapped to the node.

        :math:`(f_{ext})_i = \sum_p N_i(x_p) f_{ext}`

        Arguments
        ---------
        particles: diffmpm.particle.Particles
            Particles to map to the nodal values.
        """

        def _step(pid, args):
            f_ext, pf_ext, mapped_pos, el_nodes = args
            f_ext = f_ext.at[el_nodes[pid]].add(mapped_pos[pid] @ pf_ext[pid])
            return f_ext, pf_ext, mapped_pos, el_nodes

        mapped_positions = self.shapefn(particles.reference_loc)
        mapped_nodes = vmap(self.id_to_node_ids)(particles.element_ids).squeeze(
            -1
        )
        args = (
            self.nodes.f_ext,
            particles.f_ext,
            mapped_positions,
            mapped_nodes,
        )
        self.nodes.f_ext, _, _, _ = lax.fori_loop(
            0, len(particles), _step, args
        )

    def compute_body_force(self, particles, gravity: float):
        r"""
        Update the nodal external force based on particle mass.

        The nodal force is updated as a sum of particle body
        force for all particles mapped to th

        :math:`(f_{b})_i = \sum_p N_i(x_p) m_p g`

        Arguments
        ---------
        particles: diffmpm.particle.Particles
            Particles to map to the nodal values.
        """

        def _step(pid, args):
            f_ext, pmass, mapped_pos, el_nodes, gravity = args
            f_ext = f_ext.at[el_nodes[pid]].add(
                mapped_pos[pid] @ (pmass.T * gravity)
            )
            return f_ext, pmass, mapped_pos, el_nodes, gravity

        mapped_positions = self.shapefn(particles.reference_loc)
        mapped_nodes = vmap(self.id_to_node_ids)(particles.element_ids).squeeze(
            -1
        )
        args = (
            self.nodes.f_ext,
            particles.mass,
            mapped_positions,
            mapped_nodes,
            gravity,
        )
        self.nodes.f_ext, _, _, _, _ = lax.fori_loop(
            0, len(particles), _step, args
        )

    def compute_internal_force(self, particles):
        r"""
        Update the nodal internal force based on particle mass.

        The nodal force is updated as a sum of internal forces for
        all particles mapped to the node.

        :math:`(f_{int})_i = -\sum_p V_p * stress_p * \nabla N_i(x_p)`

        Arguments
        ---------
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
            f_int = f_int.at[el_nodes[pid]].set(update.T[..., jnp.newaxis])
            return (
                f_int,
                pvol,
                mapped_grads,
                el_nodes,
                pstress,
            )

        mapped_nodes = vmap(self.id_to_node_ids)(particles.element_ids).squeeze(
            -1
        )
        mapped_coords = vmap(self.id_to_node_loc)(
            particles.element_ids
        ).squeeze(2)
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
        self.nodes.f_int, _, _, _, _ = lax.fori_loop(
            0, len(particles), _step, args
        )

    def update_nodal_momentum(self, particles, dt: float, *args):
        """Update the nodal momentum based on total force on nodes."""
        total_force = self.nodes.get_total_force()
        self.nodes.acceleration = self.nodes.acceleration.at[:].set(
            jnp.divide(total_force, self.nodes.mass)
        )
        self.nodes.velocity = self.nodes.velocity.at[:].add(
            self.nodes.acceleration * dt
        )
        self.nodes.momentum = self.nodes.momentum.at[:].add(total_force * dt)

    def apply_boundary_constraints(self, *args):
        """Apply boundary conditions for nodal velocity."""
        self.nodes.velocity = self.nodes.velocity.at[self.boundary_nodes].set(0)
        self.nodes.momentum = self.nodes.momentum.at[self.boundary_nodes].set(0)
        self.nodes.acceleration = self.nodes.acceleration.at[
            self.boundary_nodes
        ].set(0)

    def apply_force_boundary_constraints(self, *args):
        """Apply boundary conditions for nodal forces."""
        self.nodes.f_int = self.nodes.f_int.at[self.boundary_nodes].set(0)
        self.nodes.f_ext = self.nodes.f_ext.at[self.boundary_nodes].set(0)
        self.nodes.f_damp = self.nodes.f_damp.at[self.boundary_nodes].set(0)


@register_pytree_node_class
class Linear1D(_Element):
    """
    Container for 1D line elements (and nodes).

    Element ID:            0     1     2     3
    Mesh:               +-----+-----+-----+-----+
    Node IDs:           0     1     2     3     4

    + : Nodes
    +-----+ : An element
    """

    def __init__(
        self,
        nelements: int,
        el_len: float,
        boundary_nodes: Sequence,
        nodes: Nodes = None,
    ):
        """Initialize Linear1D.

        Arguments
        ---------
        nelements : int
            Number of elements.
        el_len : float
            Length of each element.
        boundary_nodes : Sequence
            IDs of nodes that are supposed to be fixed (boundary).
        """
        self.nelements = nelements
        self.ids = jnp.arange(nelements)
        self.el_len = el_len
        if nodes is None:
            self.nodes = Nodes(
                nelements + 1,
                jnp.arange(nelements + 1).reshape(-1, 1, 1) * el_len,
            )
        else:
            self.nodes = nodes

        self.boundary_nodes = boundary_nodes

    def id_to_node_ids(self, id: int):
        """
        Node IDs corresponding to element `id`.

        Arguments
        ---------
        id : int
            Element ID.

        Returns
        -------
        jax.numpy.ndarray
            Nodal IDs of the element. Shape of returned
        array is (2, 1)
        """
        return jnp.array([id, id + 1]).reshape(2, 1)

    def id_to_node_loc(self, id: int):
        """
        Node locations corresponding to element `id`.

        Arguments
        ---------
        id : int
            Element ID.

        Returns
        -------
        jax.numpy.ndarray
            Nodal locations for the element. Shape of returned
        array is (len(id), 2, 1)
        """
        result = self.nodes.loc[jnp.array([id, id + 1])]
        return result

    def id_to_node_vel(self, id: int):
        """
        Node velocities corresponding to element `id`.

        Arguments
        ---------
        id : int
            Element ID.

        Returns
        -------
        jax.numpy.ndarray
            Nodal velocities for the element. Shape of returned
        array is (2, 1)
        """
        return self.nodes.velocity[jnp.array([id, id + 1])].reshape(2, 1)

    def shapefn(self, xi: float | jnp.ndarray):
        """
        Evaluate linear shape function.

        Arguments
        ---------
        xi : float, array_like
            Locations of particles in natural coordinates to evaluate
        the function at. Expected shape is (npoints, 1, ndim)

        Returns
        -------
        array_like
            Evaluated shape function values. The shape of the returned
        array will depend on the input shape. For example, in the linear
        case, if the input is a scalar, the returned array will be of
        the shape (1, 2, 1) but if the input is a vector then the output will
        be of the shape (len(x), 2, 1).
        """
        if len(xi.shape) != 3:
            raise ValueError(
                f"`xi` should be of size (npoints, 1, ndim); found {xi.shape}"
            )
        result = jnp.array([0.5 * (1 - xi), 0.5 * (1 + xi)]).transpose(
            1, 0, 2, 3
        )
        return result

    def _shapefn_natural_grad(self, xi: float | jnp.ndarray):
        """
        Calculate the gradient of shape function.

        This calculation is done in the natural coordinates.

        Arguments
        ---------
        x : float, array_like
            Locations of particles in natural coordinates to evaluate
        the function at.

        Returns
        -------
        array_like
            Evaluated gradient values of the shape function. The shape of
        the returned array will depend on the input shape. For example,
        in the linear case, if the input is a scalar, the returned array
        will be of the shape (1, 2) but if the input is a vector then the
        output will be of the shape (len(x), 2).
        """
        result = vmap(jacobian(self.shapefn))(xi[..., jnp.newaxis]).squeeze()

        # TODO: The following code tries to evaluate vmap even if
        # the predicate condition is true, not sure why.
        # result = lax.cond(
        #     jnp.isscalar(x),
        #     jacobian(self.shapefn),
        #     vmap(jacobian(self.shapefn)),
        #     xi
        # )
        return result.reshape(-1, 2)

    def shapefn_grad(self, xi: float | jnp.ndarray, coords: jnp.ndarray):
        """
        Gradient of shape function in physical coordinates.

        Arguments
        ---------
        xi : float, array_like
            Locations of particles to evaluate in natural coordinates.
        Expected shape (npoints, 1, ndim).
        coords : array_like
            Nodal coordinates to transform by. Expected shape
        (npoints, 1, ndim)

        Returns
        -------
        array_like
            Gradient of the shape function in physical coordinates at `xi`
        """
        if len(xi.shape) != 3:
            raise ValueError(
                f"`x` should be of size (npoints, 1, ndim); found {xi.shape}"
            )
        grad_sf = self._shapefn_natural_grad(xi)
        _jacobian = grad_sf @ coords

        result = grad_sf.T @ jnp.linalg.inv(_jacobian)
        return result.T

    def set_particle_element_ids(self, particles):
        """
        Set the element IDs for the particles.

        If the particle doesn't lie between the boundaries of any
        element, it sets the element index to -1.
        """

        @jit
        def f(x):
            idl = (
                len(self.nodes.loc)
                - 1
                - jnp.asarray(self.nodes.loc[::-1] <= x).nonzero(
                    size=1, fill_value=-1
                )[0][-1]
            )
            idg = (
                jnp.asarray(self.nodes.loc > x).nonzero(size=1, fill_value=-1)[
                    0
                ][0]
                - 1
            )
            return (idl, idg)

        ids = vmap(f)(particles.loc)
        particles.element_ids = jnp.where(
            ids[0] == ids[1], ids[0], jnp.ones_like(ids[0]) * -1
        )


class Quadrilateral4Node(_Element):
    """
    Container for 2D quadrilateral elements with 4 nodes.

    Nodes and elements are numbered as

                 15 0---0---0---0---0 19
                    | 8 | 9 | 10| 11|
                 10 0---0---0---0---0 14
                    | 4 | 5 | 6 | 7 |
                  5 0---0---0---0---0 9
                    | 0 | 1 | 2 | 3 |
                    0---0---0---0---0
                    0   1   2   3   4

    + : Nodes
    +---+
    |   | : An element
    +---+
    """

    def __init__(self, nelements: Tuple[int, int], el_len: Tuple[float, float]):
        """Initialize Quadrilateral4Node.

        Arguments
        ---------
        nelements : (int, int)
            Number of elements in X and Y direction.
        el_len : (float, float)
            Length of each element in X and Y direction.
        """
        self.nelements: jnp.ndarray = jnp.asarray(nelements)
        el_len = jnp.asarray(nelements)
        total_elements = jnp.product(nelements)
        self.ids: jnp.ndarray = jnp.arange(total_elements)

        total_nodes = jnp.product(nelements + 1)
        coords = jnp.asarray(
            list(
                itertools.product(
                    jnp.arange(nelements[1] + 1), jnp.arange(nelements[0] + 1)
                )
            )
        )
        node_locations = jnp.asarray([coords[:, 1], coords[:, 0]]).T * el_len
        self.nodes: Nodes = Nodes(total_nodes, node_locations)

    def id_to_node_ids(self, id: int):
        """
        Node IDs corresponding to element `id`.

            3----2
            |    |
            0----1

        Node ids are returned in the order as shown in the figure.

        Arguments
        ---------
        id : int
            Element ID.

        Returns
        -------
        jax.numpy.ndarray
            Nodal IDs of the element. Shape of returned
        array is (4, 2)
        """
        lower_left = (id // self.nelements[0]) * (
            self.nelements[0] + 1
        ) + id % self.nelements[0]
        return jnp.asarray(
            [
                lower_left,
                lower_left + 1,
                lower_left + self.nelements[0] + 2,
                lower_left + self.nelements[0] + 1,
            ]
        )

    def id_to_node_loc(self, id: int):
        """
        Node locations corresponding to element `id`.

        Arguments
        ---------
        id : int
            Element ID.

        Returns
        -------
        jax.numpy.ndarray
            Nodal locations for the element. Shape of returned
        array is (4, 2)
        """
        node_ids = self.id_to_node_ids(id)
        return self.nodes.loc[node_ids]

    def id_to_node_vel(self, id: int):
        """
        Node velocities corresponding to element `id`.

        Arguments
        ---------
        id : int
            Element ID.

        Returns
        -------
        jax.numpy.ndarray
            Nodal velocities for the element. Shape of returned
        array is (4, 2)
        """
        node_ids = self.id_to_node_ids(id)
        return self.nodes.vel[node_ids]


if __name__ == "__main__":
    from diffmpm.utils import _show_example

    _show_example(Linear1D(2, 1, jnp.array([0])))
