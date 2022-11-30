import matplotlib.pyplot as plt
from src.AppWorkbench.classical_monte_carlo import get_random_direction, monte_carlo_integral_from_samples, \
    get_samples_values_from_samples

from src.AppWorkbench.setup import *
from src.common.gaussian_process.covariance_functions.sobolev import Sobolev
from src.common.gaussian_process.gaussian_process import GP


def show_error_plot(result):
    for k in range(len(methods_label)):
        method = methods_label[k]
        plt.plot(ns_vector, result[:, k], label=method[0], marker=method[1])
    plt.legend()
    plt.show()


if __name__ == "__main__":
    results = np.zeros((len(ns_vector), n_methods, n_runs))  # Matrix of average error
    for run in range(n_runs):
        print(f'run {run}/{n_runs}')
        for k, ns in enumerate(ns_vector):
            sample_positions = [get_random_direction(pdf) for i in range(ns)]

            #CMC
            #print(f'Computing estimates using {ns} samples')
            abs_error_cmc = abs(ground_truth - monte_carlo_integral_from_samples(integrand, pdf, sample_positions))
            results[k, 0, run] = abs_error_cmc

            #BMC
            #print(f'Computing estimates using {ns} samples')
            gaussian_process = GP(cov_func=Sobolev(), p_func=Constant(1))
            sample_values = get_samples_values_from_samples(sample_positions, integrand)
            estimate_bmc = gaussian_process.compute_integral_BMC(sample_positions, sample_values)
            abs_error_bmc = abs(ground_truth - estimate_bmc)
            results[k, 1, run] = abs_error_bmc

            #CMC IS
            sample_positions_is = [get_random_direction(importance_sampling_pdf) for i in range(ns)]
            abs_error_icmc = abs(ground_truth - monte_carlo_integral_from_samples(integrand, importance_sampling_pdf, sample_positions))
            results[k, 2, run] = abs_error_icmc

            #BMC IS
            gaussian_process = GP(cov_func=Sobolev(), p_func=cosine_term)
            sample_values_is = get_samples_values_from_samples(sample_positions_is, integrand)
            estimate_bmc = gaussian_process.compute_integral_BMC(sample_positions_is, sample_values_is)
            abs_error_ibmc = abs(ground_truth - estimate_bmc)
            results[k, 3, run] = abs_error_ibmc




    show_error_plot(results.mean(2))
