import json
import tomllib as tl

import jax.numpy as jnp
import numpy as np

from diffmpm import element as mpel
from diffmpm import material as mpmat
from diffmpm import mesh as mpmesh
from diffmpm.constraint import Constraint
from diffmpm.node import Nodes
from diffmpm.particle import Particles


class Config:
    def __init__(self, filepath):
        self._filepath = filepath
        self.config = {}
        self.parse()

    def parse(self):
        with open(self._filepath, "rb") as f:
            self._fileconfig = tl.load(f)

        self._parse_meta(self._fileconfig)
        self._parse_output(self._fileconfig)
        self._parse_materials(self._fileconfig)
        self._parse_particles(self._fileconfig)
        mesh = self._parse_mesh(self._fileconfig)
        return mesh

    def _parse_meta(self, config):
        self.config["meta"] = config["meta"]

    def _parse_output(self, config):
        self.config["output"] = config["output"]

    def _parse_materials(self, config):
        materials = []
        for mat_config in config["materials"]:
            mat_type = mat_config.pop("type")
            mat_cls = getattr(mpmat, mat_type)
            mat = mat_cls(mat_config)
            materials.append(mat)
        self.config["materials"] = materials

    def _parse_particles(self, config):
        particle_sets = []
        for pset_config in config["particles"]:
            pmat = self.config["materials"][pset_config["material_id"]]
            with open(pset_config["file"], "r") as f:
                ploc = jnp.asarray(json.load(f))
            peids = jnp.zeros(ploc.shape[0], dtype=jnp.int32)
            pset = Particles(ploc, pmat, peids)
            pset.velocity = pset.velocity.at[:].set(
                pset_config["init_velocity"]
            )
            particle_sets.append(pset)
        self.config["particles"] = particle_sets

    def _parse_mesh(self, config):
        element_cls = getattr(mpel, config["mesh"]["element"])
        mesh_cls = getattr(mpmesh, f"Mesh{config['meta']['dimension']}D")
        constraints = [
            (jnp.asarray(c["node_ids"]), Constraint(c["dir"], c["velocity"]))
            for c in config["mesh"]["constraints"]
        ]
        if config["mesh"]["type"] == "file":
            nodes_loc = jnp.asarray(np.loadtxt(config["mesh"]["file"]))
            # nodes = Nodes(len(nodes_loc), nodes_loc)
            # elements = element_cls(nelements, el_len, boundary_nodes)
        elif config["mesh"]["type"] == "generator":
            elements = element_cls(
                config["mesh"]["nelements"],
                config["mesh"]["element_length"],
                constraints,
            )
        self.config["elements"] = elements
        mesh = mesh_cls(self.config)
        return mesh
