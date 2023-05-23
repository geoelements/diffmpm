import itertools
from typing import Sequence

import jax.numpy as jnp
from jax import jit, lax, vmap
from jax.tree_util import register_pytree_node_class
from jax_tqdm import loop_tqdm
from tqdm import tqdm

from diffmpm.element import _Element
from diffmpm.node import Nodes
from diffmpm.shapefn import Linear1DShapeFn, Linear4NodeQuad


class _MeshBase:
    def __init__(self, config: dict):
        """Initialize mesh using configuration."""
        self.particles: Sequence = config["particles"]
        self.elements: _Element = config["elements"]

    # TODO: Convert to using jax directives for loop
    def apply_on_elements(self, function, args=()):
        f = getattr(self.elements, function)
        for particle_set in self.particles:
            f(particle_set, *args)

    # TODO: Convert to using jax directives for loop
    def apply_on_particles(self, function, args=()):
        for particle_set in self.particles:
            f = getattr(particle_set, function)
            f(self.elements, *args)


class Mesh1D:
    """
    1D Mesh class with nodes, elements, and particles.
    """

    def __init__(
        self,
        nelements,
        material,
        domain_size,
        boundary_nodes,
        *,
        ppe=1,
        particle_distribution="uniform",
        elements=None,
        nodes=None,
        particles=None,
        shapefn=None,
        dim=1,
    ):
        """
        Construct a 1D Mesh.

        Arguments
        ---------
        nelements : int
            Number of elements in the mesh.
        material : diffmpm.material.Material
            Material to meshed.
        domain_size : float
            The size of the domain in consideration.
        boundary_nodes : array_like
            Node ids of boundary nodes of the mesh. Needs to be a JAX
        array.
        ppe : int
            Number of particles per element in Mesh.
        """
        self.dim = dim
        self.material = material
        self.shapefn = (
            Linear1DShapeFn(self.dim)
            if (
                shapefn is None
                or type(shapefn) is object
                or isinstance(shapefn, Mesh1D)
            )
            else shapefn
        )
        self.domain_size = domain_size
        self.nelements = nelements
        self.element_length = domain_size / nelements
        self.elements = jnp.arange(nelements) if elements is None else elements
        nnodes = nelements + 1
        self.nodes = (
            Nodes(
                nnodes,
                jnp.arange(nelements + 1) * self.element_length,
                jnp.zeros(nnodes),
                jnp.zeros(nnodes),
                jnp.zeros(nnodes),
                jnp.zeros(nnodes),
                jnp.zeros(nnodes),
                jnp.zeros(nnodes),
            )
            if (
                nodes is None
                or type(nodes) is object
                or isinstance(nodes, Mesh1D)
            )
            else nodes
        )
        self.boundary_nodes = boundary_nodes
        self.ppe = ppe
        self.particles = (
            self._init_particles(particle_distribution)
            if (
                particles is None
                or type(particles) is object
                or isinstance(particles, Mesh1D)
            )
            else particles
        )
        return


class Mesh2D:
    """
    2D Mesh class with nodes, elements, and particles.
    """

    def __init__(
        self,
        nelements,
        material,
        domain_size,
        boundary_nodes,
        *,
        ppe=1,
        particle_distribution="uniform",
        elements=None,
        nodes=None,
        particles=None,
        shapefn=None,
        dim=1,
    ):
        """
        Construct a 2D Mesh using 4-Node Quadrilateral Elements.

        Nodes and elements are numbered as

                    0---0---0---0---0
                    | 8 | 9 | 10| 11|
                 10 0---0---0---0---0
                    | 4 | 5 | 6 | 7 |
                  5 0---0---0---0---0 9
                    | 0 | 1 | 2 | 3 |
                    0---0---0---0---0
                    0   1   2   3   4


        Arguments
        ---------
        nelements : array_like
            Number of elements in the mesh in the x and y direction.
        material : diffmpm.material.Material
            Material to meshed.
        domain_size : 4-tuple, array_like
            The boundaries of the domain. Should be of the form
        (x_min, x_max, y_min, y_max)
        boundary_nodes : array_like
            Node ids of boundary nodes of the mesh. Needs to be a JAX
        array.
        ppe : int
            Number of particles per element in Mesh.
        """
        self.dim = 2
        self.material = material
        self.shapefn = (
            Linear4NodeQuad()
            if (
                shapefn is None
                or type(shapefn) is object
                or isinstance(shapefn, Mesh1D)
            )
            else shapefn
        )
        self.domain_size = domain_size
        self.nelements = jnp.asarray(nelements)
        self.element_length = jnp.array(
            [
                (domain_size[1] - domain_size[0]) / nelements[0],
                (domain_size[3] - domain_size[2]) / nelements[1],
            ]
        )
        self.elements = (
            jnp.arange(self.nelements[0] * self.nelements[1])
            if elements is None
            else elements
        )
        nnodes = jnp.product(self.nelements + 1)
        coords = jnp.asarray(
            list(
                itertools.product(
                    jnp.arange(nelements[1] + 1), jnp.arange(nelements[0] + 1)
                )
            )
        )
        node_positions = (
            jnp.asarray([coords[:, 1], coords[:, 0]]).T * self.element_length
        )

        self.nodes = (
            Nodes(
                nnodes,
                node_positions,
                jnp.zeros((nnodes, 2)),
                jnp.zeros(nnodes),
                jnp.zeros((nnodes, 2)),
                jnp.zeros((nnodes, 2)),
                jnp.zeros((nnodes, 2)),
                jnp.zeros((nnodes, 2)),
            )
            if (
                nodes is None
                or type(nodes) is object
                or isinstance(nodes, Mesh1D)
            )
            else nodes
        )
        self.boundary_nodes = boundary_nodes
        self.ppe = ppe
        self.particles = particles
        return
