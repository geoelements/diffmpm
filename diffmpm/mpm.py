from pathlib import Path

import diffmpm.writers as writers
from diffmpm.io import Config
from diffmpm.solver import MPMExplicit


class MPM:
    def __init__(self, filepath):
        self._config = Config(filepath)
        mesh = self._config.parse()
        out_dir = Path(self._config.parsed_config["output"]["folder"]).joinpath(
            self._config.parsed_config["meta"]["title"],
        )

        write_format = self._config.parsed_config["output"].get("format", None)
        if write_format is None or write_format.lower() == "none":
            writer_func = None
        elif write_format == "npz":
            writer_func = writers.NPZWriter().write
        else:
            raise ValueError(f"Specified output format not supported: {write_format}")

        if self._config.parsed_config["meta"]["type"] == "MPMExplicit":
            self.solver = MPMExplicit(
                mesh,
                self._config.parsed_config["meta"]["dt"],
                velocity_update=self._config.parsed_config["meta"]["velocity_update"],
                sim_steps=self._config.parsed_config["meta"]["nsteps"],
                out_steps=self._config.parsed_config["output"]["step_frequency"],
                out_dir=out_dir,
                writer_func=writer_func,
            )
        else:
            raise ValueError("Wrong type of solver specified.")

    def solve(self):
        """Solve the MPM simulation using JIT solver."""
        arrays = self.solver.solve_jit(
            self._config.parsed_config["external_loading"]["gravity"],
        )
        return arrays
