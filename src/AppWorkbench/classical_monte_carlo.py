from random import uniform

import numpy as np


def monte_carlo_integral(integrand, pdf, ns):
    return compute_estimate_cmc(get_samples_values_and_probabilites(integrand, pdf, ns))


def monte_carlo_integral_from_samples(integrand, pdf, samples):
    return compute_estimate_cmc(get_samples_values_and_probabilites_from_samples(integrand, pdf, samples))


# Given a set of sample values of an integrand, as well as their corresponding probabilities, #
# this function returns the classical Monte Carlo (cmc) estimate of the integral.               #
def compute_estimate_cmc(sample_values_and_probabilities):
    N = len(sample_values_and_probabilities)
    I = sum(sample_values_and_probabilities[j][0] / sample_values_and_probabilities[j][1] for j in range(N))
    return I / N


def get_samples_values_and_probabilites(integrand, pdf, ns):
    return [get_value_and_probability(get_random_direction(pdf), integrand, pdf) for i in range(ns)]


def get_samples_values_and_probabilites_from_samples(integrand, pdf, samples):
    return [get_value_and_probability(sample, integrand, pdf) for sample in samples]


def get_samples_values_from_samples(samples, integrand):
    return [(np.prod([component.eval(omega_i) for component in integrand])) for omega_i in samples]


def get_random_direction(pdf):
    u1 = uniform(0, 1)
    u2 = uniform(0, 1)
    return pdf.generate_dir(u1, u2)


def get_value_and_probability(omega_i, integrand, pdf):
    return np.prod([component.eval(omega_i) for component in integrand]), pdf.get_val(omega_i)
