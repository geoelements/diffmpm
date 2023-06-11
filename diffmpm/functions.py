import abc
import jax.numpy as jnp


class Function(abc.ABC):
    def __init__(self, id):
        self.id = id

    @abc.abstractmethod
    def value(self):
        ...


class Unit(Function):
    def __init__(self, id):
        super().__init__(id)

    def value(self, x):
        return 1.0


class Linear(Function):
    def __init__(self, id, xvalues, fxvalues):
        self.xvalues = xvalues
        self.fxvalues = fxvalues
        super().__init__(id)

    def value(self, x):
        return jnp.interp(x, self.xvalues, self.fxvalues)
