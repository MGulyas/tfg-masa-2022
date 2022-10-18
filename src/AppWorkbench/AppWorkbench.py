from random import uniform
import matplotlib.pyplot as plt
from src.AppWorkbench.setup import *

# Given a set of sample values of an integrand, as well as their corresponding probabilities, #
# this function returns the classical Monte Carlo (cmc) estimate of the integral.               #
def compute_estimate_cmc(sample_values_and_probabilites):
    N = len(sample_values_and_probabilites)
    I = sum(sample_values_and_probabilites[j][0]/sample_values_and_probabilites[j][1] for j in range(N))
    return I/N

def get_samples_values_and_probabilites(ns):
    return [get_value_and_probability(get_random_direction()) for i in range(ns)]

def get_random_direction():
    u1 = uniform(0, 1)
    u2 = uniform(0, 1)
    return pdf.generate_dir(u1, u2)

def get_value_and_probability(omega_i):
    return (np.prod([component.eval(omega_i) for component in integrand]), pdf.get_val(omega_i))

# Create a plot with the average error for each method, as a function of the number of used samples #
def show_error_plot():
    for k in range(len(methods_label)):
        method = methods_label[k]
        plt.plot(ns_vector, results[:, k], label=method[0], marker=method[1])
    plt.legend()
    plt.show()

if __name__ == "__main__":
    for k, ns in enumerate(ns_vector):
        print(f'Computing estimates using {ns} samples')
        estimate_cmc = compute_estimate_cmc(get_samples_values_and_probabilites(ns))
        abs_error = abs(ground_truth - estimate_cmc)
        results[k, 0] = abs_error

    show_error_plot()
