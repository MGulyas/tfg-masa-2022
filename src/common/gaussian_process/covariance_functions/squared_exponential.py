from src.common.gaussian_process.covariance_functions.covariance_function import CovarianceFunction
from src.common.vector_3d import Length
from math import exp


# Squared Exponential (SE) Covariance function
# [Spherical Gaussian Framework paper (Marques et al. 2013, IEEE TVCG)]
class SquaredExponential(CovarianceFunction):
    def __init__(self, l, noise):
        super().__init__(noise)
        self.l = l  # Length-scale parameter (controls the smoothness of the GP model)

    def eval(self, omega_i, omega_j):
        r = Length(omega_i - omega_j)  # Euclidean distance between the samples
        return exp(-(r ** 2) / (2 * self.l ** 2))
