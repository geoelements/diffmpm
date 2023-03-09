class Material:
    """
    Base material class.
    """

    def __init__(self, E, density):
        """
        Initialize material properties.

        Arguments
        ---------
        E : float
            Young's modulus of the material.
        density : float
            Density of the material.
        """
        self.E = E
        self.density = density
