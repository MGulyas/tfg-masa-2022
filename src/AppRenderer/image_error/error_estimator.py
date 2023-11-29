import csv
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm

from src.AppRenderer.integrators.classical_monte_carlo_integrator import ClassicalMonteCarloIntegrator
from src.AppRenderer.integrators.improved_bayesian_monte_carlo_integrator import ImprovedBayesianMonteCarloIntegrator
from src.AppRenderer.scenes.classical_monte_carlo_scene import classical_monte_carlo_scene


def estimate_numerical_error(reference_path, result_path):
    reference = np.load(reference_path)
    result = np.load(result_path)
    shape = reference.shape

    sum_of_errors = 0
    if result.shape != shape:
        raise Exception(f'Image shape {result.shape} does not match with reference shape {shape}')

    for i in range(shape[0]):
        for j in range(shape[1]):
            sum_of_errors += np.linalg.norm(reference[i, j] - result[i, j])
    return sum_of_errors / (shape[0] * shape[1])


if __name__ == "__main__":
    DIRECTORY = 'src/out/images_for_numerical_error_estimation/improved_bayesian_monte_carlo/1_gp/30_samples'
    n_runs = 20
    n_samples = 30
    scene = classical_monte_carlo_scene(use_env_map=True)
    header = ['error', 'runtime']
    with open('src/out/images_for_numerical_error_estimation/improved_bayesian_monte_carlo/1_gp/30_samples/error.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(n_runs):
            print(f'Run {i + 1}/{n_runs}')
            start_time = time.time()
            integrator = ImprovedBayesianMonteCarloIntegrator(n_samples, DIRECTORY + f'/run_{i}', n_gps=1)
            integrator.add_scene(scene)
            integrator.prerender()
            integrator.render()
            data = [estimate_numerical_error("src/AppRenderer/image_error/reference_image"
                                             "/rendered_image_MC_3000_samples.npy", DIRECTORY +
                                             f"/run_{i}_improved_BMC_30_samples.npy"),
                    time.time() - start_time]
            writer.writerow(data)
