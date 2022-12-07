import torch

def Linear1DShapefn(xi):
    """
    Linear 1D shape function in natural coordinates
    """
    shapefn = torch.zeros(2)
    shapefn[0] = 0.5 * (1 - xi)
    shapefn[1] = 0.5 * (1 + xi)
    return shapefn


def Linear1DGradsf(xi):
    """
    Linear 1D gradient of shape function in natural coordinates.
    """
    gradsf = torch.zeros()
    gradsf[0] = -0.5
    gradsf[1] = 0.5
    return gradsf