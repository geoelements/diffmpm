from collections import namedtuple

import jax.numpy as jnp
from diffmpm.constraint import Constraint
from diffmpm.element import Quadrilateral4Node
from diffmpm.functions import Linear
from diffmpm.material import LinearElastic, SimpleMaterial
from diffmpm.mesh import Mesh2D
from diffmpm.particle import Particles
from diffmpm.solver import MPMExplicit
from diffmpm.forces import NodalForce

particles = Particles(
    jnp.array([[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]).reshape(
        4, 1, 2
    ),
    # SimpleMaterial({"E": 1, "density": 1}),
    LinearElastic({"density": 1, "youngs_modulus": 1000, "poisson_ratio": 0.0}),
    jnp.array([0, 0, 0, 0]),
)
particles.velocity += jnp.array([[0.0, 0.0]])
cons = [
    (jnp.array([0, 1]), Constraint(1, 0.0)),
    (jnp.array([2, 3]), Constraint(1, -0.01)),
]

mathfn = Linear(0, jnp.array([0.0, 0.5, 1.0]), jnp.array([0.0, 1.0, 1.0]))
cnf = [NodalForce(jnp.array([0, 1]), mathfn, 1, 10)]
elements = Quadrilateral4Node((1, 1), (1.0, 1.0), cons, concentrated_nodal_forces=cnf)
elements.nodes.velocity = elements.nodes.velocity.at[:, :, 0].set(1)
# elements.nodes.mass += 1
# elements.compute_external_force(particles)
# print(elements.nodes.f_ext.squeeze())
# elements.compute_body_force(particles, jnp.array([0, -10]).reshape(1, 2))
# elements.compute_internal_force(particles)
mesh = Mesh2D({"elements": elements, "particles": [particles]})
mpm = MPMExplicit(mesh, 0.01, scheme="usf")
result = mpm.solve(10, 0)
print(result)
