import numpy as np

from src.common.constants import HUGEVALUE, EPSILON


class Ray:
    # Initializer
    def __init__(self, origin=np.zeros(3),
                 direction=np.zeros(3), tmax=HUGEVALUE):
        self.o = origin
        self.d = direction
        self.t_max = tmax
        self.t_min = EPSILON

    # Member Functions
    def get_hitpoint(self, t):
        return self.o + self.d * t