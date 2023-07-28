import abc
import logging
from pathlib import Path
from pyevtk.hl import pointsToVTK

from typing import Tuple, Annotated, Any
from jax.typing import ArrayLike
import numpy as np

logger = logging.getLogger(__file__)

__all__ = ["_Writer", "EmptyWriter", "NPZWriter"]


class _Writer(abc.ABC):
    """Base writer class."""

    @abc.abstractmethod
    def write(self):
        ...


class EmptyWriter(_Writer):
    """Empty writer used when output is not to be written."""

    def write(self, args, transforms, **kwargs):
        """Empty function."""
        pass


class NPZWriter(_Writer):
    """Writer to write output in `.npz` format."""

    def write(
        self,
        args: Tuple[
            Annotated[ArrayLike, "JAX arrays to be written"],
            Annotated[int, "step number of the simulation"],
        ],
        transforms: Any,
        **kwargs,
    ):
        """Writes the output arrays as `.npz` files."""
        arrays, step = args
        max_digits = int(np.log10(kwargs["max_steps"])) + 1
        if step == 0:
            req_zeros = max_digits - 1
        else:
            req_zeros = max_digits - (int(np.log10(step)) + 1)
        fileno = f"{'0' * req_zeros}{step}"
        filepath = Path(kwargs["out_dir"]).joinpath(f"particles_{fileno}.npz")
        if not filepath.parent.is_dir():
            filepath.parent.mkdir(parents=True)
        np.savez(filepath, **arrays)
        logger.info(f"Saved particle data for step {step} at {filepath}")


class VTKWriter(Writer):
    def write(self, args, transforms, **kwargs):
        arrays, step = args
        max_digits = int(np.log10(kwargs["max_steps"])) + 1
        if step == 0:
            req_zeros = max_digits - 1
        else:
            req_zeros = max_digits - (int(np.log10(step)) + 1)
        fileno = f"{'0' * req_zeros}{step}"
        filepath = Path(kwargs["out_dir"]).joinpath(f"particles_{fileno}")
        if not filepath.parent.is_dir():
            filepath.parent.mkdir(parents=True)
        coords = np.array(arrays["loc"])
        x = coords[:, :, 0].flatten()
        y = coords[:, :, 1].flatten()
        z = np.zeros_like(x)
        strain = np.array(arrays["strain"])
        stress = np.array(arrays["stress"])
        velocity = np.array(arrays["velocity"])
        velocity_x = velocity[:, :, 0].flatten()
        veloctiy_y = velocity[:, :, 1].flatten()
        velocity_z = np.zeros_like(velocity_x)
        strain_xx = strain[:, :, 0].flatten()
        strain_yy = strain[:, :, 1].flatten()
        strain_zz = strain[:, :, 2].flatten()
        strain_xy = strain[:, :, 3].flatten()
        strain_xz = strain[:, :, 4].flatten()
        strain_yz = strain[:, :, 5].flatten()
        stress_xx = stress[:, :, 0].flatten()
        stress_yy = stress[:, :, 1].flatten()
        stress_zz = stress[:, :, 2].flatten()
        stress_xy = stress[:, :, 3].flatten()
        stress_xz = stress[:, :, 4].flatten()
        stress_yz = stress[:, :, 5].flatten()
        x = coords[:, :, 0].flatten()
        y = coords[:, :, 1].flatten()
        z = np.zeros_like(x)
        pointsToVTK(
            f"{filepath}",
            np.array(x),
            np.array(y),
            z,
            data={
                "strain_xx": strain_xx,
                "strain_yy": strain_yy,
                "strain_zz": strain_zz,
                "strain_xy": strain_xy,
                "strain_xz": strain_xz,
                "strain_yz": strain_yz,
                "stress_xx": stress_xx,
                "stress_yy": stress_yy,
                "stress_zz": stress_zz,
                "stress_xy": stress_xy,
                "stress_xz": stress_xz,
                "stress_yz": stress_yz,
                "velocity_x": velocity_x,
                "velocity_y": veloctiy_y,
                "velocity_z": velocity_z,
            },
        )
        logger.info(f"Saved particle data for step {step} at {filepath}")
