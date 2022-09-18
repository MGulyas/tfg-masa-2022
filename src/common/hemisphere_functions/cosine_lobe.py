from math import pi

from src.common.hemisphere_functions.function import Function
from src.common.vector_3d import Vector3D, Dot


class CosineLobe(Function):
    def __init__(self, exp_):
        self.exp = exp_
        integral = self.get_integral()
        super().__init__(integral)

    def eval(self, omega_i):
        normal = Vector3D(0, 1, 0)
        return Dot(normal, omega_i) ** self.exp

    def get_integral(self):
        return 2 * pi / (self.exp + 1)