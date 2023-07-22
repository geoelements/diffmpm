import abc
import logging
from pathlib import Path
from typing import Annotated, Any, Tuple

import numpy as np
from jax.typing import ArrayLike

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
