from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class Constraint:
    """Generic velocity constraints to apply on nodes or particles."""

    def __init__(self, dir: int, velocity: float):
        """Contains 2 govering parameters.

        Attributes
        ----------
        dir : int
            Direction in which constraint is applied.
        velocity : float
            Constrained velocity to be applied.
        """
        self.dir = dir
        self.velocity = velocity

    def tree_flatten(self):
        return ((), (self.dir, self.velocity))

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        return cls(*aux_data)

    def apply_vel(self, obj, ids):
        """Apply constraint values to the passed object.

        Parameters
        ----------
        obj : diffmpm.node.Nodes, diffmpm.particle._ParticlesState
            Object on which the constraint is applied
        ids : array_like
            The indices of the container `obj` on which the constraint
            will be applied.
        """
        velocity = obj.velocity.at[ids, :, self.dir].set(self.velocity)
        return velocity

    def apply_mom(self, obj, ids):
        """Apply constraint values to the passed object.

        Parameters
        ----------
        obj : diffmpm.node.Nodes, diffmpm.particle._ParticlesState
            Object on which the constraint is applied
        ids : array_like
            The indices of the container `obj` on which the constraint
            will be applied.
        """
        momentum = obj.momentum.at[ids, :, self.dir].set(
            obj.mass[ids, :, 0] * self.velocity
        )
        return momentum

    def apply_acc(self, obj, ids):
        """Apply constraint values to the passed object.

        Parameters
        ----------
        obj : diffmpm.node.Nodes, diffmpm.particle._ParticlesState
            Object on which the constraint is applied
        ids : array_like
            The indices of the container `obj` on which the constraint
            will be applied.
        """
        acceleration = obj.acceleration.at[ids, :, self.dir].set(0)
        return acceleration
