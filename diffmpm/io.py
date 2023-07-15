import json
import tomllib as tl

import jax.numpy as jnp

from diffmpm import element as mpel
from diffmpm import materials as mpmat
from diffmpm import mesh as mpmesh
from diffmpm.constraint import Constraint
from diffmpm.forces import NodalForce, ParticleTraction
from diffmpm.functions import Linear, Unit
from diffmpm.particle import Particles


class Config:
    def __init__(self, filepath):
        self._filepath = filepath
        self.parsed_config = {}
        self.parse()

    def parse(self):
        with open(self._filepath, "rb") as f:
            self._fileconfig = tl.load(f)

        self.entity_sets = json.load(open(self._fileconfig["mesh"]["entity_sets"]))
        self._parse_meta(self._fileconfig)
        self._parse_output(self._fileconfig)
        self._parse_materials(self._fileconfig)
        self._parse_particles(self._fileconfig)
        if "math_functions" in self._fileconfig:
            self._parse_math_functions(self._fileconfig)
        self._parse_external_loading(self._fileconfig)
        mesh = self._parse_mesh(self._fileconfig)
        return mesh

    def _get_node_set_ids(self, set_ids):
        all_ids = []
        for set_id in set_ids:
            ids = self.entity_sets["node_sets"][str(set_id)]
            all_ids.extend(ids)
        return jnp.asarray(all_ids)

    def _get_particle_set_ids(self, set_type, set_nos, set_ids):
        all_ids = []
        for set_no in set_nos:
            for set_id in set_ids:
                ids = self.entity_sets["particle_sets"][set_no][str(set_id)]
                all_ids.extend(ids)
        return jnp.asarray(all_ids)

    def _parse_meta(self, config):
        self.parsed_config["meta"] = config["meta"]

    def _parse_output(self, config):
        self.parsed_config["output"] = config["output"]

    def _parse_materials(self, config):
        materials = []
        for mat_config in config["materials"]:
            mat_type = mat_config.pop("type")
            mat_cls = getattr(mpmat, mat_type)
            mat = mat_cls(mat_config)
            materials.append(mat)
        self.parsed_config["materials"] = materials

    def _parse_particles(self, config):
        particle_sets = []
        for pset_config in config["particles"]:
            pmat = self.parsed_config["materials"][pset_config["material_id"]]
            with open(pset_config["file"], "r") as f:
                ploc = jnp.asarray(json.load(f))
            peids = jnp.zeros(ploc.shape[0], dtype=jnp.int32)
            pset = Particles(ploc, pmat, peids)
            pset.velocity = pset.velocity.at[:].set(pset_config["init_velocity"])
            particle_sets.append(pset)
        self.parsed_config["particles"] = particle_sets

    def _parse_math_functions(self, config):
        flist = []
        for i, fnconfig in enumerate(config["math_functions"]):
            if fnconfig["type"] == "Linear":
                fn = Linear(
                    i,
                    jnp.array(fnconfig["xvalues"]),
                    jnp.array(fnconfig["fxvalues"]),
                )
                flist.append(fn)
            else:
                raise NotImplementedError(
                    "Function type other than `Linear` not yet supported"
                )
        self.parsed_config["math_functions"] = flist

    def _parse_external_loading(self, config):
        external_loading = {}
        external_loading["gravity"] = jnp.array(config["external_loading"]["gravity"])
        external_loading["concentrated_nodal_forces"] = []
        particle_surface_traction = []
        if "concentrated_nodal_forces" in config["external_loading"]:
            cnf_list = []
            for cnfconfig in config["external_loading"]["concentrated_nodal_forces"]:
                if "math_function_id" in cnfconfig:
                    fn = self.parsed_config["math_functions"][
                        cnfconfig["math_function_id"]
                    ]
                else:
                    fn = Unit(-1)
                cnf = NodalForce(
                    node_ids=self._get_node_set_ids(cnfconfig["nset_ids"]),
                    function=fn,
                    dir=cnfconfig["dir"],
                    force=cnfconfig["force"],
                )
                cnf_list.append(cnf)
            external_loading["concentrated_nodal_forces"] = cnf_list

        if "particle_surface_traction" in config["external_loading"]:
            pst_list = []
            for pstconfig in config["external_loading"]["particle_surface_traction"]:
                if "math_function_id" in pstconfig:
                    fn = self.parsed_config["math_functions"][
                        pstconfig["math_function_id"]
                    ]
                else:
                    fn = Unit(-1)
                pst = ParticleTraction(
                    pset=pstconfig["pset"],
                    pids=self._get_particle_set_ids(
                        "particle", pstconfig["pset"], pstconfig["pset_ids"]
                    ),
                    function=fn,
                    dir=pstconfig["dir"],
                    traction=pstconfig["traction"],
                )
                pst_list.append(pst)
            particle_surface_traction.extend(pst_list)
        self.parsed_config["external_loading"] = external_loading
        self.parsed_config["particle_surface_traction"] = particle_surface_traction

    def _parse_mesh(self, config):
        element_cls = getattr(mpel, config["mesh"]["element"])
        mesh_cls = getattr(mpmesh, f"Mesh{config['meta']['dimension']}D")

        constraints = [
            (
                self._get_node_set_ids(c["nset_ids"]),
                Constraint(c["dir"], c["velocity"]),
            )
            for c in config["mesh"]["constraints"]
        ]

        if config["mesh"]["type"] == "generator":
            elements = element_cls(
                config["mesh"]["nelements"],
                jnp.prod(jnp.array(config["mesh"]["nelements"])),
                config["mesh"]["element_length"],
                constraints,
                concentrated_nodal_forces=self.parsed_config["external_loading"][
                    "concentrated_nodal_forces"
                ],
            )
        else:
            raise NotImplementedError(
                "Mesh type other than `generator` not yet supported."
            )

        self.parsed_config["elements"] = elements
        mesh = mesh_cls(self.parsed_config)
        return mesh
