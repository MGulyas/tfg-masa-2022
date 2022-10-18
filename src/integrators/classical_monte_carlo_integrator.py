from random import uniform
import numpy as np

from src.common.color import RGBColor, BLACK
from src.common.functions import oriented_hemi_dir, Vector3D
from src.common.hemisphere_functions.constant import Constant
from src.common.hemisphere_functions.cosine_lobe import CosineLobe
from src.common.pdfs.uniform_pdf import UniformPDF
from src.common.ray import Ray
from src.integrators.integrator import Integrator
import itertools


class ClassicalMonteCarloIntegrator(Integrator):  # Classic Monte Carlo Integrator

    def __init__(self, n, filename_, experiment_name=''):
        filename_mc = filename_ + '_MC_' + str(n) + '_samples' + experiment_name
        super().__init__(filename_mc)
        self.n_samples = n
        self.pdf = UniformPDF()
        self.cosine_term = CosineLobe(1)

    def compute_estimate_cmc(self, sample_values_and_probabilites):
        N = len(sample_values_and_probabilites)
        cmc_estimates_by_color = []
        for i in range(3):
            I = sum(sample_values_and_probabilites[j][0][i] / sample_values_and_probabilites[j][1] for j in range(N)) #TODO make better
            cmc_estimates_by_color.append(I/N)
        return RGBColor(cmc_estimates_by_color[0], cmc_estimates_by_color[1], cmc_estimates_by_color[2])

    def get_samples_values_and_probabilites(self, scene, hit_data, brdf):
        return [self.get_value_and_probability(scene, hit_data, self.get_random_direction(hit_data), brdf) for i in range(self.n_samples)]

    def get_random_direction(self, hit_data):
        u1 = uniform(0, 1)  # random number between 0 and 1
        u2 = uniform(0, 1)
        return oriented_hemi_dir(self.pdf, u1, u2, hit_data.normal)

    def get_value_and_probability(self, scene, hit_data, omega_i, brdf):
        def evaluate_integrand(integrand):
            return np.prod([component.eval(omega_i) for component in integrand])
        l_i = self.get_intensity_from_direction(scene, hit_data, omega_i)
        values_by_channel = list(map(evaluate_integrand, zip(l_i, brdf, itertools.repeat(self.cosine_term))))
        return values_by_channel, self.pdf.get_val(omega_i)

    def get_intensity_from_direction(self, scene, hit_data, omega_i):
        new_ray = Ray(origin=hit_data.hit_point, direction=omega_i)
        new_hit_data = scene.closest_hit(new_ray)
        kd = scene.object_list[new_hit_data.primitive_index].get_BRDF().kd
        return (Constant(kd.r), Constant(kd.g), Constant(kd.b))

    def compute_color(self, ray, scene):
        hit_data = scene.closest_hit(ray)
        if hit_data.has_hit:
            kd = scene.object_list[hit_data.primitive_index].get_BRDF().kd
            brdf = (Constant(kd.r), Constant(kd.g), Constant(kd.b))
            return self.compute_estimate_cmc(self.get_samples_values_and_probabilites(scene, hit_data, brdf))
        return BLACK
        # TODO do environment maps later
