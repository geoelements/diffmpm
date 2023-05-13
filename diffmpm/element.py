import abc
import itertools
from typing import Tuple

import jax.numpy as jnp
from jax import jacobian, vmap, lax

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

    # @abc.abstractmethod
    # def tree_flatten(self):
    #     ...

    # @classmethod
    # @abc.abstractmethod
    # def tree_unflatten(self):
    #     ...


class Linear1D(_Element):
    """
    Container for 1D line elements (and nodes).

    Element ID:            0     1     2     3
    Mesh:               +-----+-----+-----+-----+
    Node IDs:           0     1     2     3     4

    + : Nodes
    +-----+ : An element
    """

    def __init__(self, nelements: int, el_len: float):
        """Initialize Linear1D.

        Arguments
        ---------
        nelements : int
            Number of elements.
        el_len : float
            Length of each element.
        """
        self.ids: jnp.ndarray = jnp.arange(nelements)
        self.el_len: float = el_len
        self.nodes: Nodes = Nodes(
            nelements + 1, jnp.arange(nelements + 1).reshape(-1, 1) * el_len
        )

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
        return self.nodes.loc[jnp.array([id, id + 1])].transpose(1, 0, 2)

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

    def shapefn(self, xi):
        """
        Evaluate linear shape function.

        Arguments
        ---------
        xi : float, array_like
            Locations of particles in natural coordinates to evaluate
        the function at.

        Returns
        -------
        array_like
            Evaluated shape function values. The shape of the returned
        array will depend on the input shape. For example, in the linear
        case, if the input is a scalar, the returned array will be of
        the shape (1, 2, 1) but if the input is a vector then the output will
        be of the shape (len(x), 2, 1).
        """
        result = jnp.array([0.5 * (1 - xi), 0.5 * (1 + xi)]).reshape(-1, 2, 1)
        return result

    def _shapefn_natural_grad(self, xi):
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
        if jnp.isscalar(xi):
            result = jacobian(self.shapefn)(xi)
        else:
            result = vmap(jacobian(self.shapefn))(xi.reshape(-1, 1)).squeeze()

        # TODO: The following code tries to evaluate vmap even if
        # the predicate condition is true, not sure why.
        # result = lax.cond(
        #     jnp.isscalar(x),
        #     jacobian(self.shapefn),
        #     vmap(jacobian(self.shapefn)),
        #     xi
        # )
        return result.reshape(-1, 2)

    def shapefn_grad(self, x, coords):
        """
        Gradient of shape function in physical coordinates.

        Arguments
        ---------
        x : float, array_like
            Locations of particles to evaluate in natural coordinates.
        coords : array_like
            Nodal coordinates to transform by.

        Returns
        -------
        array_like
            Gradient of the shape function in physical coordinates at `x`
        """
        length = abs(coords[1] - coords[0])
        result = self._shapefn_natural_grad(x) * 2 / length
        return result

    def update_nodal_mass(self, particles):
        r"""
        Update the nodal mass based on particle mass.

        The nodal mass is updated as a sum of particle mass for
        all particles mapped to the node.

        :math:`(m)_i = \sum_p N_i(x_p) m_p`
        """

        def _step(pid, args):
            pmass, mass, mapped_pos, el_nodes = args
            mass = mass.at[el_nodes[pid]].add(pmass[pid] * mapped_pos[pid])
            return pmass, mass, mapped_pos, el_nodes

        mapped_positions = self.shapefn(particles.reference_loc)
        mapped_nodes = vmap(self.id_to_node_ids)(
            particles.element_ids
        ).squeeze()
        args = (
            particles.mass,
            self.nodes.mass,
            mapped_positions,
            mapped_nodes,
        )
        _, self.nodes.mass, _, _ = lax.fori_loop(0, len(particles), _step, args)

    def update_particle_natural_coords(self, particles):
        """
        Update natural coordinates for the particles.

        Whenever the particles' physical coordinates change, their
        natural coordinates need to be updated. This function updates
        the natural coordinates of the particles based on the element
        a particle is a part of. The update formula is

        :math:`xi = (x - x_{n_0}) 2 / l - 1`

        If a particle is not in any element (element_id = -1), its
        natural coordinate is set to 0.

        Arguments
        ---------
        particles: diffmpm.particle.Particles
            Particles whose natural coordinates need to be updated based
        on these elements.
        """
        t = self.id_to_node_loc(particles.element_ids)
        xi_coords = (particles.loc - (t[:, 0, ...] + t[:, 1, ...]) / 2) * (
            2 / (t[:, 1, ...] - t[:, 0, ...])
        )
        particles.reference_loc = xi_coords


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
