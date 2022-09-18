from math import pi

from src.common.hemisphere_functions.function import Function


class Constant(Function):
    def __init__(self, const_value_):
        self.const_value = const_value_
        integral = self.get_integral()
        super().__init__(integral)

    def eval(self, omega_i):
        return self.const_value

    def get_integral(self):
        return 2 * pi * self.const_value