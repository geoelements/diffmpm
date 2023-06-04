from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class Constraint:
    def __init__(self, dir, velocity):
        self.dir = dir
        self.velocity = velocity

    def tree_flatten(self):
        return ((), (self.dir, self.velocity))

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        return cls(*aux_data)

    def apply(self, obj, ids):
        """
        Apply constraint values to the passed object.

        Arguments
        ---------
        obj : diffmpm.node.Nodes, diffmpm.particle.Particles
            Object on which the constraint is applied
        ids : array_like
            The indices of the container `obj` on which the constraint
        will be applied.
        """
        obj.velocity = obj.velocity.at[ids, :, self.dir].set(self.velocity)
        obj.momentum = obj.momentum.at[ids, :, self.dir].set(
            obj.mass[ids, :, 0] * self.velocity
        )
        obj.acceleration = obj.acceleration.at[ids, :, self.dir].set(0)
