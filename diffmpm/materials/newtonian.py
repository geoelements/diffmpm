import jax.numpy as jnp
from jax import Array, lax
from jax.typing import ArrayLike

from ._base import _Material


class Newtonian(_Material):
    """Newtonian fluid material model."""

    _props = ("density", "bulk_modulus", "dynamic_viscosity")
    state_vars = ("pressure",)

    def __init__(self, material_properties: dict):
        """Create a Newtonian material.

        Parameters
        ----------
        material_properties: dict
            Dictionary with material properties. For newtonian
        materials, `density`, `bulk_modulus` and `dynamic_viscosity`
        are required keys.
        """
        self.validate_props(material_properties)
        compressibility = 1

        if material_properties.get("incompressible", False):
            compressibility = 0

        self.properties = {
            **material_properties,
            "compressibility": compressibility,
        }

    def __repr__(self):
        return f"Newtonian(props={self.properties})"

    def initialize_state_variables(self, nparticles: int) -> dict:
        """Return initial state variables dictionary.

        Parameters
        ----------
        nparticles : int
            Number of particles being simulated with this material.

        Returns
        -------
        dict
            Dictionary of state variables initialized with values
            decided by material type.
        """
        state_vars_dict = {var: jnp.zeros((nparticles, 1)) for var in self.state_vars}
        return state_vars_dict

    def _thermodynamic_pressure(self, volumetric_strain: ArrayLike) -> Array:
        return -self.properties["bulk_modulus"] * volumetric_strain

    def compute_stress(self, particles):
        """Compute material stress."""
        ndim = particles.loc.shape[-1]
        if ndim not in {2, 3}:
            raise ValueError(f"Cannot compute stress for {ndim}-d Newotonian material.")
        volumetric_strain_rate = (
            particles.strain_rate[:, 0] + particles.strain_rate[:, 1]
        )
        particles.state_vars["pressure"] = (
            particles.state_vars["pressure"]
            .at[:]
            .add(
                self.properties["compressibility"]
                * self._thermodynamic_pressure(particles.dvolumetric_strain)
            )
        )

        volumetric_stress_component = self.properties["compressibility"] * (
            -particles.state_vars["pressure"]
            - (2 * self.properties["dynamic_viscosity"] * volumetric_strain_rate / 3)
        )

        stress = jnp.zeros_like(particles.stress)
        stress = stress.at[:, 0].set(
            volumetric_stress_component
            + 2 * self.properties["dynamic_viscosity"] * particles.strain_rate[:, 0]
        )
        stress = stress.at[:, 1].set(
            volumetric_stress_component
            + 2 * self.properties["dynamic_viscosity"] * particles.strain_rate[:, 1]
        )

        extra_component_2 = lax.select(
            ndim == 3,
            2 * self.properties["dynamic_viscosity"] * particles.strain_rate[:, 2],
            jnp.zeros_like(particles.strain_rate[:, 2]),
        )
        stress = stress.at[:, 2].set(volumetric_stress_component + extra_component_2)

        stress = stress.at[:, 3].set(
            self.properties["dynamic_viscosity"] * particles.strain_rate[:, 3]
        )

        component_4 = lax.select(
            ndim == 3,
            self.properties["dynamic_viscosity"] * particles.strain_rate[:, 4],
            jnp.zeros_like(particles.strain_rate[:, 4]),
        )
        stress = stress.at[:, 4].set(component_4)
        component_5 = lax.select(
            ndim == 3,
            self.properties["dynamic_viscosity"] * particles.strain_rate[:, 5],
            jnp.zeros_like(particles.strain_rate[:, 5]),
        )
        stress = stress.at[:, 5].set(component_5)

        return stress
