import torch

class Node1D:
    """1D MPM node
    Attributes:
        id: Index of node class.
        x: Location of node.
        mass: Mass at node.
        velocity: Velocity at node.
        momentum: Momentum at node.
        f_int: Internal force.
        f_ext: External force.
    """

    def __init__(self):
        self.id = None
        self.x = torch.tensor(0)
        self.mass = torch.tensor(0)
        self.velocity = torch.tensor(0)
        self.momentum = torch.tensor(0)
        self.f_int = torch.tensor(0)
        self.f_ext = torch.tensor(0)

