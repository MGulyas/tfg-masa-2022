from src.common.gaussian_process.covariance_functions.covariance_function import CovarianceFunction
from src.common.vector_3d import Length


# Sobolev Covariance function
# [Optimal Sample Weights paper (Marques et al. 2019, Computer Graphics Forum)]
class Sobolev(CovarianceFunction):
    def __init__(self, s=1.4):
        self.s = s  # Smoothness parameter (controls the smoothness of the GP model)

    def eval(self, omega_i, omega_j):
        s = self.s
        r = Length(omega_i - omega_j)  # Euclidean distance between the samples
        return (2 ** (2 * s - 1)) / s - r ** (2 * s - 2)


class SobolevFixedValue():
    def __init__(self, omega_i, s=1.4):
        self.s = s  # Smoothness parameter (controls the smoothness of the GP model)
        self.omega_i = omega_i

    def eval(self, omega_j):
        s = self.s
        r = Length(self.omega_i - omega_j)  # Euclidean distance between the samples
        return (2 ** (2 * s - 1)) / s - r ** (2 * s - 2)
