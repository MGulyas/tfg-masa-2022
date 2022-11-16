from random import uniform
import numpy as np

from src.AppRenderer.integrators.integrator import Integrator
from src.AppWorkbench.classical_monte_carlo import get_random_direction
from src.common.color import BLACK
from src.common.functions import oriented_hemi_dir
from src.common.gaussian_process.covariance_functions.sobolev import Sobolev
from src.common.gaussian_process.gaussian_process import GP
from src.common.hemisphere_functions.constant import Constant
from src.common.hemisphere_functions.cosine_lobe import CosineLobe
from src.common.pdfs.uniform_pdf import UniformPDF
from src.common.ray import Ray


class BayesianMonteCarloIntegrator(Integrator):
    def __init__(self, n, myGP, filename_, experiment_name=''):
        filename_bmc = filename_ + '_BMC_' + str(n) + '_samples' + experiment_name
        super().__init__(filename_bmc)
        self.n_samples = n
        self.myGP = myGP
        self.pdf = UniformPDF()
        self.cosine_term = CosineLobe(1)

    def get_sample_values(self, scene, hit_data, brdf):
        return [self.get_value(scene, hit_data, self.get_random_direction(hit_data), brdf) for i in range(self.n_samples)]

    def get_random_direction(self, hit_data):
        u1 = uniform(0, 1)  # random number between 0 and 1
        u2 = uniform(0, 1)
        return oriented_hemi_dir(self.pdf, u1, u2, hit_data.normal)

    def get_value(self, scene, hit_data, omega_i, brdf):
        def evaluate_integrand(integrand):
            return np.prod([component.eval(omega_i) for component in integrand])
        l_i = self.get_incident_radiance_from_direction(scene, hit_data, omega_i)
        values_by_channel = list(map(evaluate_integrand, zip(l_i, brdf, np.itertools.repeat(self.cosine_term))))
        return values_by_channel

    def get_incident_radiance_from_direction(self, scene, hit_data, omega_i):
        new_ray = Ray(origin=hit_data.hit_point, direction=omega_i)
        new_hit_data = scene.closest_hit(new_ray)
        if new_hit_data.has_hit:
            incident_radiance = scene.object_list[new_hit_data.primitive_index].emission
        else:
            if scene.env_map:
                incident_radiance = scene.env_map.getValue(new_ray.d)
            else:
                incident_radiance = BLACK
        return (Constant(incident_radiance.r), Constant(incident_radiance.g), Constant(incident_radiance.b))


    def compute_color(self, ray, scene):
        hit_data = scene.closest_hit(ray)
        if hit_data.has_hit:
            gaussian_process = GP(cov_func=Sobolev(), p_func=Constant(1))
            kd = scene.object_list[hit_data.primitive_index].get_BRDF().kd
            brdf = (Constant(kd.r), Constant(kd.g), Constant(kd.b))
            sample_positions = [get_random_direction(self.pdf) for i in range(self.n_samples)]
            return gaussian_process.compute_integral_BMC(sample_positions, self.get_sample_values(sample_positions))
        if scene.env_map:
            return scene.env_map.getValue(ray.d)
        return BLACK