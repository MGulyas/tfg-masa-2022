from src.common.constants import HUGEVALUE, EPSILON
from src.common.vector_3d import Vector3D


class Ray:
    # Initializer
    def __init__(self, origin=Vector3D(0, 0, 0),
                 direction=Vector3D(0, 0, 0), tmax=HUGEVALUE):
        self.o = origin
        self.d = direction
        self.t_max = tmax
        self.t_min = EPSILON

    # Member Functions
    def get_hitpoint(self, t):
        return self.o + self.d * t