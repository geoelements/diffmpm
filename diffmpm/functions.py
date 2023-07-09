import abc

import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class


class Function(abc.ABC):
    def __init__(self, id):
        self.id = id

    @abc.abstractmethod
    def value(self):
        ...


@register_pytree_node_class
class Unit(Function):
    def __init__(self, id):
        super().__init__(id)

    def value(self, x):
        return 1.0

    def tree_flatten(self):
        return ((), (self.id))

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        return cls(*aux_data)


@register_pytree_node_class
class Linear(Function):
    def __init__(self, id, xvalues, fxvalues):
        self.xvalues = xvalues
        self.fxvalues = fxvalues
        super().__init__(id)

    def value(self, x):
        return jnp.interp(x, self.xvalues, self.fxvalues)

    def tree_flatten(self):
        return ((), (self.id, self.xvalues, self.fxvalues))

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        return cls(*aux_data)
