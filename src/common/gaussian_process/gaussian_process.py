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

        self.original_normal = None  # normal vector of hemisphere when weights were computed
        self.weights = None

    # Method which computes and inverts the covariance matrix Q
    #  - IMPORTANT: requires that the samples positions are already known
    def _compute_inv_Q(self, sample_positions):
        n = len(sample_positions)
        Q = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                Q[i, j] = self.cov_func.eval(sample_positions[i], sample_positions[j])
                if i == j:
                    Q[i, j] += self.noise
        return np.linalg.inv(Q)

    # Method in charge of computing the z vector.
    #  - IMPORTANT: requires that the samples positions are already known
    def _compute_z(self, sample_positions, ns=10000):
        # The z vector z  = [z_1, z_2, ..., z_n] is a vector of integrals, where the value of each element is computed
        #  based on the position omega_n of the nth sample. In most of the cases, these integrals do not have an
        #  analytic solution. Therefore, we can resort to classic Monte Carlo to estimate the value of these integrals
        #  (that is, of each element z_i of z).
        return np.array([monte_carlo_integral([SobolevFixedValue(sample_positions[j]), self.p_func], pdf, ns) for j in
                         range(len(sample_positions))])

    # Method in charge of computing the BMC integral estimate (assuming the the prior mean function has value 0)
    #todo, te cambié constant_prior que era una lista por una sola constante. Quizá no sea lo que querías
    def compute_integral_BMC(self, samples, values, constant_prior=None):
        constant_prior = constant_prior if constant_prior is not None else 0.0
        # Inverted covariance matrix Q^{-1}
        invQ = self._compute_inv_Q(samples)
        # Vector of z coefficients (see theory slides and Practice 3 text for more details)
        z = self._compute_z(samples)[np.newaxis]
        # Sample weights. Contains the value by which each sample y_i must be multiplied to compute the BMC estimate.
        weights = z @ invQ
        I_prior = constant_prior * self.p_func.get_integral()
        y_prior = (np.array([constant_prior] * len(values), dtype=float)[np.newaxis]).T
        values = np.array([values], dtype=float).T
        return I_prior + weights @ (values - y_prior)

    def compute_weights(self, sample_positions):
        invQ = self._compute_inv_Q(sample_positions)
        z = self._compute_z(sample_positions)
        self.weights = z @ invQ

    def compute_integral_BMC_with_fixed_samples(self, values, constant_prior=[0, 0, 0]):
        I_prior = [channel * self.p_func.get_integral() for channel in constant_prior]
        y_prior = np.array([constant_prior] * len(values))
        values = np.array(values)  # values = np.array([values]).T workbench?
        return I_prior + self.weights @ np.array(values - y_prior)
