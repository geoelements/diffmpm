import abc
import itertools
from typing import Sequence

import jax.numpy as jnp

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
        self.nodes: Nodes = Nodes(
            nelements + 1, jnp.arange(nelements + 1) * el_len
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
        array is (2, 1)
        """
        return self.nodes.loc[jnp.array([id, id + 1])].reshape(2, 1)

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

    def __init__(
        self, nelements: Sequence[int, int], el_len: Sequence[float, float]
    ):
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
