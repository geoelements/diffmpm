import abc
from functools import partial
from typing import Callable, Sequence, Tuple

import jax.numpy as jnp
from jax import lax, jit, tree_util
from jax.tree_util import register_pytree_node_class, tree_map

from diffmpm.element import _ElementState
import diffmpm.element as dfel
from diffmpm.node import _NodesState
from diffmpm.particle import _ParticlesState
import diffmpm.particle as dpart
from diffmpm.forces import ParticleTraction

__all__ = ["_MeshBase", "Mesh1D", "Mesh2D"]


class _MeshBase(abc.ABC):
    """Base class for Meshes.

    .. note::
        If attributes other than elements and particles are added
        then the child class should also implement `tree_flatten` and
        `tree_unflatten` correctly or that information will get lost.
    """

    ndim: int

    def __init__(self, config: dict):
        """Initialize mesh using configuration."""
        self.particles: Sequence[_ParticlesState] = config["particles"]
        self.elements: _ElementState = config["elements"]
        self.particle_tractions = config["particle_surface_traction"]
        self.elementor = config["elementor"]

    # TODO: Change to allow called functions to return outputs
    def apply_on_elements(self, function: str, args: Tuple = ()):
        """Apply a given function to elements.

        Parameters
        ----------
        function: str
            A string corresponding to a function name in `_Element`.
        args: tuple
            Parameters to be passed to the function.
        """
        f = getattr(self.elementor, function)

        def _func(particles, *, func, elements, fargs):
            return func(elements, particles, *fargs)

        partial_func = partial(_func, func=f, elements=self.elements, fargs=args)
        _out = tree_map(
            partial_func,
            self.particles,
            is_leaf=lambda x: isinstance(x, _ParticlesState),
        )
        if function == "set_particle_element_ids":
            self.particles = _out
        elif function == "apply_boundary_constraints":
            self.elements.nodes = _out[0]
        elif function == "compute_volume":
            self.elements = _out[0]
        elif _out[0] is not None:
            _temp = tree_util.tree_transpose(
                tree_util.tree_structure([0 for e in _out]),
                tree_util.tree_structure(_out[0]),
                _out,
            )

            def reduce_attr(state_1, state_2, *, orig):
                new_val = state_1 + state_2 - orig
                return new_val

            attr = _temp[1][0]
            p_reduce_attr = partial(
                reduce_attr,
                attr=attr,
                orig=getattr(self.elements.nodes, attr),
            )
            new_val = tree_util.tree_reduce(
                p_reduce_attr, _temp[0], is_leaf=lambda x: isinstance(x, _NodesState)
            )
            self.elements.nodes = self.elements.nodes.replace(**{attr: new_val})

    def apply_on_particles(self, function: str, args: Tuple = ()):
        """Apply a given function to particles.

        Parameters
        ----------
        function: str
            A string corresponding to a function name in `_ParticlesState`.
        args: tuple
            Parameters to be passed to the function.
        """

        def _func(particles, *, elements, fname, fargs):
            f = getattr(dpart, fname)
            return f(particles, elements, *fargs)

        partial_func = partial(
            _func, elements=self.elements, fname=function, fargs=(self.elementor, *args)
        )
        new_states = tree_map(
            partial_func,
            self.particles,
            is_leaf=lambda x: isinstance(x, _ParticlesState),
        )
        self.particles = new_states

    def apply_traction_on_particles(self, curr_time: float):
        """Apply tractions on particles.

        Parameters
        ----------
        curr_time: float
            Current time in the simulation.
        """
        self.apply_on_particles("zero_traction")

        def func(ptraction, *, particle_sets):
            def f(particles, *, ptraction, traction_val):
                return dpart.assign_traction(
                    particles, ptraction.pids, ptraction.dir, traction_val
                )

            factor = ptraction.function.value(curr_time)
            traction_val = factor * ptraction.traction
            partial_f = partial(f, ptraction=ptraction, traction_val=traction_val)
            traction_sets = tree_map(
                partial_f,
                particle_sets,
                is_leaf=lambda x: isinstance(x, _ParticlesState),
            )
            return tuple(traction_sets)

        partial_func = partial(func, particle_sets=self.particles)
        if self.particle_tractions:
            _out = tree_map(
                partial_func,
                self.particle_tractions,
                is_leaf=lambda x: isinstance(x, ParticleTraction),
            )
            _temp = tree_util.tree_transpose(
                tree_util.tree_structure([0 for e in _out]),
                tree_util.tree_structure(_out[0]),
                _out,
            )
            tractions_ = tree_util.tree_reduce(
                lambda x, y: x + y, _temp, is_leaf=lambda x: isinstance(x, list)
            )
            self.particles = [
                pset.replace(traction=traction)
                for pset, traction in zip(self.particles, tractions_)
            ]

            self.apply_on_elements("apply_particle_traction_forces")

    def tree_flatten(self):
        children = (self.particles, self.elements)
        aux_data = (self.elementor, self.particle_tractions)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            {
                "particles": children[0],
                "elements": children[1],
                "elementor": aux_data[0],
                "particle_surface_traction": aux_data[1],
            }
        )


@register_pytree_node_class
class Mesh1D(_MeshBase):
    """1D Mesh class with nodes, elements, and particles."""

    def __init__(self, config: dict):
        """Initialize a 1D Mesh.

        Parameters
        ----------
        config: dict
            Configuration to be used for initialization. It _should_
            contain `elements` and `particles` keys.
        """
        self.ndim = 1
        super().__init__(config)


@register_pytree_node_class
class Mesh2D(_MeshBase):
    """1D Mesh class with nodes, elements, and particles."""

    def __init__(self, config: dict):
        """Initialize a 2D Mesh.

        Parameters
        ----------
        config: dict
            Configuration to be used for initialization. It _should_
            contain `elements` and `particles` keys.
        """
        self.ndim = 2
        super().__init__(config)
