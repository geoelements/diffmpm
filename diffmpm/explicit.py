import abc
from dataclasses import dataclass
from functools import partial
from typing import Callable, NamedTuple, Optional, Sequence

from jax import Array, vmap
from jax.tree_util import tree_map, tree_reduce, tree_structure, tree_transpose

from diffmpm.element import Quad4N, _ElementsState
from diffmpm.forces import ParticleTraction
from diffmpm.node import _reset_node_props
from diffmpm.particle import (
    _assign_traction,
    _compute_particle_volume,
    _compute_strain,
    _compute_stress,
    _get_natural_coords,
    _ParticlesState,
    _update_particle_position_velocity,
    _update_particle_volume,
    _zero_traction,
)


class MeshState(NamedTuple):
    elements: _ElementsState
    particles: _ParticlesState
    particle_tractions: Sequence[ParticleTraction]

    @classmethod
    def _apply_traction_on_particles(
        cls, particles, particle_tractions, curr_time: float
    ):
        """Apply tractions on particles.

        Parameters
        ----------
        curr_time: float
            Current time in the simulation.
        """
        pass


class Solver(abc.ABC):
    @abc.abstractmethod
    def init_state(*args, **kwargs):
        pass

    @abc.abstractmethod
    def update(*args, **kwargs):
        pass

    def run(*args, **kwargs):
        pass


def _reduce_attr(state_1, state_2, *, orig):
    new_val = state_1 + state_2 - orig
    return new_val


def _tree_transpose(pytree):
    _out = tree_transpose(
        tree_structure([0 for e in pytree]), tree_structure(pytree[0]), pytree
    )
    return _out


@dataclass(eq=False)
class ExplicitSolver(Solver):
    el_type: Quad4N
    tol: float
    dt: float
    sim_steps: int
    out_steps: int
    out_dir: str
    gravity: Array
    scheme: str = "usf"
    velocity_update: Optional[bool] = False
    writer_func: Optional[Callable] = None

    def init_state(self, config):
        elements = config["elements"]
        particles = config["particles"]
        new_peids: list = tree_map(
            self.el_type._get_particles_element_ids,
            particles,
            [elements] * len(particles),
            is_leaf=lambda x: isinstance(x, _ParticlesState)
            or isinstance(x, _ElementsState),
        )
        new_evol = self.el_type._compute_volume(elements.el_len, elements.volume)

        temp_pprops = tree_map(
            _compute_particle_volume,
            new_peids,
            [self.el_type.total_elements] * len(particles),
            [new_evol] * len(particles),
            [p.volume for p in particles],
            [p.size for p in particles],
            [p.mass for p in particles],
            [p.density for p in particles],
        )
        new_pprops = _tree_transpose(temp_pprops)
        elements = elements.replace(volume=new_evol)
        particles = [
            p.replace(
                element_ids=new_peids,
                mass=new_pprops["mass"][i],
                size=new_pprops["size"][i],
                volume=new_pprops["volume"][i],
            )
            for i, p in enumerate(particles)
        ]
        return MeshState(
            elements=elements,
            particles=particles,
            particle_tractions=config["particle_surface_traction"],
        )

    def update(self, state: MeshState, step, *args, **kwargs):
        _elements, _particles = state.elements, state.particles
        # Nodal properties that are to be reset at the beginning of the
        # update step.
        new_nmass, new_nmom, new_nfint, new_nfext, new_nfdamp = _reset_node_props(
            _elements.nodes
        )

        # New Element IDs for particles in each particle set.
        # This is a `tree_map` function so that each particle set gets
        # new EIDs.
        new_peids: list = tree_map(
            self.el_type._get_particles_element_ids,
            _particles,
            [_elements] * len(_particles),
            is_leaf=lambda x: isinstance(x, _ParticlesState)
            or isinstance(x, _ElementsState),
        )
        map_fn = vmap(self.el_type._get_mapped_nodes, (0, None))
        new_pmapped_node_ids = tree_map(
            map_fn,
            new_peids,
            [_elements.nelements[0]] * len(_particles),
            is_leaf=lambda x: isinstance(x, _ParticlesState)
            or isinstance(x, _ElementsState),
        )

        # New natural coordinates of the particles.
        # This is again a `tree_map`-ed function for each particle set.
        # The signature of the function is
        #       `get_natural_coords(particles.loc, elements)`
        # Attributes required:
        #   - Element IDs of the particles
        #   - Nodal coords of the elements corresponding to the above element ids.
        def _leaf_fn(x):
            return isinstance(x, _ParticlesState) or isinstance(x, Array)

        new_pxi = tree_map(
            _get_natural_coords,
            _particles,
            new_pmapped_node_ids,
            [_elements.nodes.loc] * len(_particles),
            is_leaf=_leaf_fn,
        )

        # New nodal mass based on particle mass
        # Required:
        #   - Nodal mass (new_nmass)
        #   - Particle natural coords (new_pxi)
        #   - Mapped nodes
        #   - Particle element IDs (new_peids) (list)
        # new_nmass = self.el_type._compute_nodal_mass(new_nmass, new_pxi, new_peids)
        temp_nmass = tree_map(
            self.el_type._compute_nodal_mass,
            [new_nmass] * len(_particles),
            [p.mass for p in _particles],
            new_pxi,
            new_peids,
            new_pmapped_node_ids,
            [p.ids for p in _particles],
            is_leaf=lambda x: isinstance(x, _ParticlesState)
            or isinstance(x, Array)
            or isinstance(x, int),
        )
        partial_reduce_attr = partial(_reduce_attr, orig=new_nmass)
        new_nmass = tree_reduce(partial_reduce_attr, temp_nmass)

        # New nodal momentum based on particle momentum
        # Required:
        #   - Nodal momentum (new_nmom)
        #   - Particle natural coords (new_pxi)
        #   - Mapped nodes
        #   - Particle element IDs (new_peids) (list)
        # new_nmom = _compute_nodal_momentum(new_nmom, new_xi, new_peids)
        temp_nmom = tree_map(
            self.el_type._compute_nodal_momentum,
            [new_nmom] * len(_particles),
            [p.mass for p in _particles],
            [p.velocity for p in _particles],
            new_pxi,
            new_peids,
            new_pmapped_node_ids,
            [p.ids for p in _particles],
            is_leaf=lambda x: isinstance(x, _ParticlesState)
            or isinstance(x, Array)
            or isinstance(x, int),
        )
        partial_reduce_attr = partial(_reduce_attr, orig=new_nmom)
        new_nmom = tree_reduce(partial_reduce_attr, temp_nmom)

        # New nodal velocity based on nodal momentum
        # Required:
        #   - Nodal mass (new_nmass)
        #   - Current nodal velocity (_elements.nodes.velocity)
        #   - Nodal momentum (new_nmom)
        #   - Tolerance (tol)
        # new_nvel = _compute_nodal_velocity(
        #     new_nmass, new_nmom, _elements.nodes.velocity, self.tol
        # )
        temp_nvel = tree_map(
            self.el_type._compute_nodal_velocity,
            new_nmass,
            new_nmom,
            _elements.nodes.velocity,
        )
        partial_reduce_attr = partial(_reduce_attr, orig=_elements.nodes.velocity)
        new_nvel = tree_reduce(partial_reduce_attr, temp_nvel)

        # Apply boundary constraints on newly calculated props.
        # Since nodal acceleration hasn't been updated yet, we
        # use the current states nodal acceleration.
        # Required:
        #   - Constraints (_elements.constraints)
        #   - Nodal velocity (new_nvel)
        #   - Nodal momentum (new_nmom)
        #   - Nodal acceleration (new_nacc)
        new_nvel, new_nmom, new_nacc = self.el_type._apply_boundary_constraints(
            new_nvel,
            new_nmom,
            _elements.nodes.acceleration,
            new_nmass,
            _elements.constraints,
        )

        if self.scheme == "usf":
            # Compute particle strain
            # Required:
            #   - Mapped node ids
            #   - Mapped node locs
            #   - Mapped node vels
            #   - Particle natural coords (new_pxi)
            #   - Current particle strains (_particles.strain)
            #   - Particles locs
            #   - Particle volumetric strains (_particles.volumetric_strain_centroid)
            _temp = tree_map(
                _compute_strain,
                [p.strain for p in _particles],
                new_pxi,
                [p.loc for p in _particles],
                [p.volumetric_strain_centroid for p in _particles],
                [p.ids for p in _particles],
                new_pmapped_node_ids,
                [_elements.nodes.loc] * len(_particles),
                [new_nvel] * len(_particles),
                [self.el_type] * len(_particles),
                [self.dt] * len(_particles),
                is_leaf=lambda x: isinstance(x, _ParticlesState)
                or isinstance(x, Quad4N)
                or isinstance(x, Array)
                or isinstance(x, float),
            )

            _strains = _tree_transpose(_temp)
            new_pstrain_rate = _strains["strain_rate"]
            new_pdstrain = _strains["dstrain"]
            new_pstrain = _strains["strain"]
            new_pdvolumetric_strain = _strains["dvolumetric_strain"]
            new_pvolumetric_strain_centroid = _strains["volumetric_strain_centroid"]

            # Compute new particle volumes based on updated strain
            # Required:
            #   - Particle volumetric dstrain (new_pdvolumetric_strain)
            # new_pvol, new_pdensity = _update_particle_volume(new_pdvolumetric_strain)
            _temp = tree_map(
                _update_particle_volume,
                [p.volume for p in _particles],
                [p.density for p in _particles],
                new_pdvolumetric_strain,
            )

            new_pvol, new_pdensity = _tree_transpose(_temp)
            # Compute particle stress
            # Required:
            #   - Particle state since different materials need different
            #     particle properties to calculate stress.
            # new_pstress = _compute_stress(_particles)
            new_pstress = tree_map(
                _compute_stress,
                [p.stress for p in _particles],
                new_pstrain,
                new_pdstrain,
                [p.material for p in _particles],
                is_leaf=lambda x: isinstance(x, _ParticlesState),
            )

        # Compute external forces on nodes
        # Required:
        #   - Nodal external forces (new_nfext)
        #   - Particle natural coords (new_pxi)
        #   - Mapped Node ids
        # new_nfext = self.el_type._compute_external_force(new_nfext, new_pxi, *args)
        temp_nfext = tree_map(
            self.el_type._compute_external_force,
            [new_nfext] * len(_particles),
            [p.f_ext for p in _particles],
            new_pxi,
            [p.ids for p in _particles],
            new_pmapped_node_ids,
        )
        partial_reduce_attr = partial(_reduce_attr, orig=new_nfext)
        new_nfext = tree_reduce(partial_reduce_attr, temp_nfext)

        # Compute body forces on nodes
        # Required:
        #   - Nodal external forces (new_nfext)
        #   - Particle natural coords (new_pxi)
        #   - Mapped Node ids
        #   - gravity
        # new_nfext = _compute_body_force(new_nfext, new_pxi, gravity, *args)
        temp_nfext = tree_map(
            self.el_type._compute_body_force,
            [new_nfext] * len(_particles),
            [p.mass for p in _particles],
            new_pxi,
            new_pmapped_node_ids,
            [p.ids for p in _particles],
            [self.gravity] * len(_particles),
        )
        partial_reduce_attr = partial(_reduce_attr, orig=new_nfext)
        new_nfext = tree_reduce(partial_reduce_attr, temp_nfext)

        # TODO: Apply traction on particles
        new_ptraction = tree_map(
            _zero_traction,
            [p.traction for p in _particles],
            is_leaf=lambda x: isinstance(x, _ParticlesState),
        )

        def func(ptract_, ptraction, pvol, psize, *, curr_time):
            def f(ptraction, pvol, psize, *, ptract_, traction_val):
                return _assign_traction(
                    ptraction, pvol, psize, ptract_.pids, ptract_.dir, traction_val
                )

            factor = ptract_.function.value(curr_time)
            traction_val = factor * ptract_.traction
            partial_f = partial(f, ptract_=ptract_, traction_val=traction_val)
            traction_sets = tree_map(
                partial_f,
                ptraction,
                pvol,
                psize,
                is_leaf=lambda x: isinstance(x, _ParticlesState),
            )
            return tuple(traction_sets)

        partial_func = partial(
            func, ptract_=state.particle_tractions, curr_time=step * self.dt
        )
        if state.particle_tractions:
            _out = tree_map(
                partial_func,
                state.particle_tractions,
                new_ptraction,
                new_pvol,
                [p.size for p in _particles],
                is_leaf=lambda x: isinstance(x, ParticleTraction)
                or isinstance(x, Array),
            )
            breakpoint()
            _temp = _tree_transpose(_out)
            new_ptraction = tree_reduce(
                lambda x, y: x + y, _temp, is_leaf=lambda x: isinstance(x, list)
            )

        # breakpoint()
        temp_nfext = tree_map(
            self.el_type._apply_particle_traction_forces,
            new_pxi,
            new_pmapped_node_ids,
            [new_nfext] * len(_particles),
            new_ptraction,
            [p.ids for p in _particles],
        )
        partial_reduce_attr = partial(_reduce_attr, orig=new_nfext)
        new_nfext = tree_reduce(partial_reduce_attr, temp_nfext)

        # Apply nodal concentrated forces
        # Required:
        #   - Concentrated forces on nodes (_elements.concentrated_nodal_forces)
        #   - Nodal external forces (new_nfext)
        #   - current time
        if _elements.concentrated_nodal_forces:
            new_nfext = self.el_type._apply_concentrated_nodal_forces(
                new_nfext, _elements.concentrated_nodal_forces, self.dt * step
            )
        # Compute internal forces on nodes
        # Required:
        #   - Mapped node ids
        #   - Mapped node locs
        #   - Nodal internal forces (new_nfint)
        #   - Particle natural coords (new_pxi)
        #   - Particle volume (new_pvol)
        #   - Particle stress (new_pstress)
        temp_nfint = tree_map(
            self.el_type._compute_internal_force,
            [new_nfint] * len(_particles),
            [_elements.nodes.loc] * len(_particles),
            new_pmapped_node_ids,
            new_pxi,
            new_pvol,
            new_pstress,
            [p.ids for p in _particles],
        )
        partial_reduce_attr = partial(_reduce_attr, orig=new_nfint)
        new_nfint = tree_reduce(partial_reduce_attr, temp_nfint)

        if self.scheme == "usl":
            # TODO: Calculate strains and stresses
            pass

        # Update nodal acceleration based on nodal forces
        # Required:
        #   - Nodal forces (new_nfint, new_nfext, new_nfdamp)
        #   - Nodal mass
        #   - Constraints (_elements.constraints)
        #   - Tolerance (self.tol)
        total_force = new_nfint + new_nfext + new_nfdamp
        new_nacc = self.el_type._update_nodal_acceleration(
            total_force, new_nacc, new_nmass, _elements.constraints, self.tol
        )
        # Update nodal acceleration based on nodal forces
        # Required:
        #   - Nodal forces (new_nfint, new_nfext, new_nfdamp)
        #   - Nodal mass
        #   - Constraints (_elements.constraints)
        #   - Tolerance (self.tol)
        new_nvel = self.el_type._update_nodal_velocity(
            total_force, new_nvel, new_nmass, _elements.constraints, self.dt, self.tol
        )

        # Update nodal momentum based on nodal forces
        # Required:
        #   - Nodal mass (new_nmass)
        #   - Nodal velocity (new_nvel)
        #   - Tolerance (self.tol)
        new_nmom = self.el_type._update_nodal_momentum(
            new_nmass, new_nvel, _elements.constraints, self.tol
        )

        # Update particle position and velocity
        # Required:
        #   - Particle natural coords (new_pxi)
        #   - Timestep (self.dt)
        #   - self.velocity_update
        #   - Mapped node ids
        #   - Mapped node vels
        #   - Mapped node accelerations
        #   - Particle locs
        _temp_new_vals = tree_map(
            _update_particle_position_velocity,
            [self.el_type] * len(_particles),
            [p.loc for p in _particles],
            [p.velocity for p in _particles],
            [p.momentum for p in _particles],
            [p.mass for p in _particles],
            new_pxi,
            new_pmapped_node_ids,
            [new_nvel] * len(_particles),
            [new_nacc] * len(_particles),
            [self.velocity_update] * len(_particles),
            [self.dt] * len(_particles),
            is_leaf=lambda x: isinstance(x, Array)
            or isinstance(x, Quad4N)
            or isinstance(x, float)
            or isinstance(x, bool),
        )
        _new_vals = _tree_transpose(_temp_new_vals)
        new_pvel = _new_vals["velocity"]
        new_ploc = _new_vals["loc"]
        new_pmom = _new_vals["momentum"]

        new_node_state = _elements.nodes.replace(
            velocity=new_nvel,
            acceleration=new_nacc,
            mass=new_nmass,
            momentum=new_nmom,
            f_int=new_nfint,
            f_ext=new_nfext,
            f_damp=new_nfdamp,
        )
        new_element_state = _elements.replace(nodes=new_node_state)
        new_particle_states = [
            _p.replace(
                loc=new_ploc[i],
                element_ids=new_peids[i],
                density=new_pdensity[i],
                volume=new_pvol[i],
                velocity=new_pvel[i],
                momentum=new_pmom[i],
                strain=new_pstrain[i],
                stress=new_pstress[i],
                strain_rate=new_pstrain_rate[i],
                dstrain=new_pdstrain[i],
                reference_loc=new_pxi[i],
                dvolumetric_strain=new_pdvolumetric_strain[i],
                volumetric_strain_centroid=new_pvolumetric_strain_centroid[i],
            )
            for i, _p in enumerate(_particles)
        ]

        new_mesh_state = MeshState(
            elements=new_element_state,
            particles=new_particle_states,
            particle_tractions=state.particle_tractions,
        )
        return new_mesh_state
