import abc
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__file__)


class Writer(abc.ABC):
    @abc.abstractmethod
    def write(self):
        ...


class EmptyWriter(Writer):
    def write(self, args, transforms, **kwargs):
        pass


class NPZWriter(Writer):
    def write(self, args, transforms, **kwargs):
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
