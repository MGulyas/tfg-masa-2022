import numpy as np


# Gaussian Process class for the unit hemisphere
from src.AppWorkbench.classical_monte_carlo import monte_carlo_integral_from_samples, monte_carlo_integral
from src.AppWorkbench.setup import pdf
from src.common.gaussian_process.covariance_functions.sobolev import SobolevFixedValue


class GP:

    # Initializer
    def __init__(self, cov_func, p_func, noise_=0.01):

        # Attribute containing the covariance function
        self.cov_func = cov_func  # k

        # Analytically-known part of the integrand, i.e., the function p(x)
        self.p_func = p_func

        # Noise term to be added in the diagonal of the covariance matrix. This typically small value is used to avoid
        #  numerical instabilities when inverting the covariance matrix Q (preempts the matrix Q from being singular or
        #  close to singular, and thus not invertible).
        self.noise = noise_


    # Method which computes and inverts the covariance matrix Q
    #  - IMPORTANT: requires that the samples positions are already known
    def compute_inv_Q(self, sample_positions):
        n = len(sample_positions)
        Q = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                Q[i, j] = 1
                Q[i, j] = self.cov_func.eval(sample_positions[i], sample_positions[j])
                if i == j:
                    Q[i, j] += self.noise
        return np.linalg.inv(Q)

    # Method in charge of computing the z vector.
    #  - IMPORTANT: requires that the samples positions are already known
    def compute_z(self, sample_positions):
        # The z vector z  = [z_1, z_2, ..., z_n] is a vector of integrals, where the value of each element is computed
        #  based on the position omega_n of the nth sample. In most of the cases, these integrals do not have an
        #  analytic solution. Therefore, we can resort to classic Monte Carlo to estimate the value of these integrals
        #  (that is, of each element z_i of z).
        ns = 250
        return np.array([monte_carlo_integral([SobolevFixedValue(sample_positions[j]), self.p_func], pdf, ns) for j in range(len(sample_positions))])

    # Method in charge of computing the BMC integral estimate (assuming the the prior mean function has value 0)
    def compute_integral_BMC(self, sample_positions, sample_values):

        # Inverted covariance matrix Q^{-1}
        invQ = self.compute_inv_Q(sample_positions)
        # Vector of z coefficients (see theory slides and Practice 3 text for more details)
        z = self.compute_z(sample_positions)
        # Sample weights. Contains the value by which each sample y_i must be multiplied to compute the BMC estimate.
        weights = z @ invQ

        return weights @ sample_values
