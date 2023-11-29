import itertools
import random
from random import uniform
import numpy as np

from src.AppRenderer.integrators.integrator import Integrator
from src.common.color import BLACK, RGBColor
from src.common.functions import oriented_hemi_dir
from src.common.gaussian_process.covariance_functions.sobolev import Sobolev
from src.common.gaussian_process.gaussian_process import GP
from src.common.hemisphere_functions.constant import Constant
from src.common.hemisphere_functions.cosine_lobe import CosineLobe
from src.common.pdfs.uniform_pdf import UniformPDF
from src.common.ray import Ray
from src.common.vector_3d import Vector3D


class WithPriorBayesianMonteCarloIntegrator(Integrator):
    def __init__(self, n,  filename_, experiment_name='', n_gps=4):
        filename_bmc = filename_ + '_improved_BMC_with_prior_' + str(n) + '_samples' + experiment_name + '_' + str(n_gps) + 'gps'
        super().__init__(filename_bmc)
        self.n_samples = n
        self.n_gps = n_gps
        self.gaussian_processes = [GP(cov_func=Sobolev(), p_func=Constant(1)) for _ in range(self.n_gps)]
        self.pdf = UniformPDF()
        self.cosine_term = CosineLobe(1)
        self.normal = Vector3D(0, 1, 0)

        self.sample_sets = [np.hstack([(self.get_random_direction().to_numpy()[np.newaxis]).T
                                  for _ in range(self.n_samples)]) for _ in range(self.n_gps)]
        self.compute_weights_on_gaussian_processes()

    def get_sample_values(self, scene, hit_data, brdf, samples):
        return [self.get_value(scene, hit_data, sample, brdf) for sample in samples]

    def compute_weights_on_gaussian_processes(self):
        for i in range(self.n_gps):
            self.gaussian_processes[i].compute_weights(self.get_column_vectors_from_matrix(self.sample_sets[i]))

    def get_random_direction(self):
        u1 = uniform(0, 1)  # random number between 0 and 1
        u2 = uniform(0, 1)
        return oriented_hemi_dir(self.pdf, u1, u2, self.normal)

    def get_value(self, scene, hit_data, omega_i, brdf):
        def evaluate_integrand(integrand):
            return np.prod([component.eval(omega_i) for component in integrand])

        l_i = self.get_incident_radiance_from_direction(scene, hit_data, omega_i)
        values_by_channel = list(
            map(evaluate_integrand, zip(l_i, brdf, itertools.repeat(CosineLobe(1, hit_data.normal)))))
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

    def are_aligned(self, n, n0):
        return np.isclose(np.dot(n0, n), 1)

    def generate_rotation_matrix(self, n):  # This method generates the rotation matrix to go from n0 to n.
        n0 = self.normal.to_numpy()
        n = n.to_numpy()
        n = n / np.linalg.norm(n)

        if self.are_aligned(n, n0):
            return np.identity(3)

        # Compute the rotation axis
        rotation_axis = np.cross(n0, n)

        # Compute the rotation angle
        rotation_angle = np.arccos(np.dot(n0, n))

        # Create the rotation matrix
        rotation_matrix = np.array([
            [np.cos(rotation_angle) + rotation_axis[0] ** 2 * (1 - np.cos(rotation_angle)),
             rotation_axis[0] * rotation_axis[1] * (1 - np.cos(rotation_angle)) - rotation_axis[2] * np.sin(
                 rotation_angle),
             rotation_axis[0] * rotation_axis[2] * (1 - np.cos(rotation_angle)) + rotation_axis[1] * np.sin(
                 rotation_angle)],
            [rotation_axis[1] * rotation_axis[0] * (1 - np.cos(rotation_angle)) + rotation_axis[2] * np.sin(
                rotation_angle),
             np.cos(rotation_angle) + rotation_axis[1] ** 2 * (1 - np.cos(rotation_angle)),
             rotation_axis[1] * rotation_axis[2] * (1 - np.cos(rotation_angle)) - rotation_axis[0] * np.sin(
                 rotation_angle)],
            [rotation_axis[2] * rotation_axis[0] * (1 - np.cos(rotation_angle)) - rotation_axis[1] * np.sin(
                rotation_angle),
             rotation_axis[2] * rotation_axis[1] * (1 - np.cos(rotation_angle)) + rotation_axis[0] * np.sin(
                 rotation_angle),
             np.cos(rotation_angle) + rotation_axis[2] ** 2 * (1 - np.cos(rotation_angle))]
        ])
        # Generate an additional random rotation on azimuth to reduce visible noise patterns
        random_rotation_matrix = self.generate_random_rotation()
        return random_rotation_matrix@rotation_matrix

    def generate_random_rotation(self, max_angle=0.1):
        phi = np.random.uniform(0, max_angle)
        cosine = np.cos(phi)
        sine = np.sin(phi)
        return np.array([[cosine, 0, -sine],
                         [0, 1, 0],
                         [sine, 0, cosine]])

    def get_column_vectors_from_matrix(self, M):
        return [Vector3D(M[0, i], M[1, i], M[2, i]) for i in range(M.shape[1])]

    def rotate_samples(self, rotation_matrix, i):
        rotated_samples_matrix = rotation_matrix @ self.sample_sets[i]
        return self.get_column_vectors_from_matrix(rotated_samples_matrix)

    def compute_color(self, ray, scene, prior=0):
        hit_data = scene.closest_hit(ray)
        if hit_data.has_hit:
            kd = scene.object_list[hit_data.primitive_index].get_BRDF().kd
            brdf = (Constant(kd.r), Constant(kd.g), Constant(kd.b))
            rotation_matrix = self.generate_rotation_matrix(hit_data.normal)

            i = random.randint(0,self.n_gps-1)
            rotated_samples = self.rotate_samples(rotation_matrix, i)
            value = self.gaussian_processes[i].compute_integral_BMC_with_fixed_samples(self.get_sample_values(scene,
                                                                                                         hit_data, brdf,
                                                                                                         rotated_samples))
            return RGBColor(value[0], value[1], value[2])
        if scene.env_map:
            return scene.env_map.getValue(ray.d)
        return BLACK

    def prerender(self):
        cam = self.scene.camera  # camera object
        print('Pre-rendering image: ' + self.get_filename())
        for x in range(0, cam.width):
            for y in range(0, cam.height):
                if x%2 == 0 and y%2 == 0:
                    ray = Ray(origin=Vector3D(0,0,0), direction=self.scene.camera.get_direction(x, y))
                    pixel = self.compute_color(ray, self.scene)
                    self.scene.set_pixel(pixel, x, y)  # save pixel to pixel array
            progress = (x / cam.width) * 100
            print('\r\tProgress pre-render: ' + str(progress) + '%', end='')
        # save image to file
        print('\r\tProgress pre-render: 100% \n\t', end='')

    def render(self):
        cam = self.scene.camera  # camera object
        print('Rendering Image: ' + self.get_filename())
        for x in range(0, cam.width):
            for y in range(0, cam.height):
                if x % 2 == 1 or y % 2 == 1:
                    ray = Ray(origin=Vector3D(0,0,0), direction=self.scene.camera.get_direction(x, y))
                    color_left = self.scene.get_pixel(x-1, y)
                    color_right = self.scene.get_pixel(x+1, y)
                    color_up = self.scene.get_pixel(x, y+1)
                    color_down = self.scene.get_pixel(x, y-1)
                    prior_guess_color = BLACK.blend([color_left, color_right, color_up, color_down])
                    pixel = self.compute_color(ray, self.scene, prior_guess_color)
                    self.scene.set_pixel(pixel, x, y)  # save pixel to pixel array
            progress = (x / cam.width) * 100
            print('\r\tProgress: ' + str(progress) + '%', end='')
        # save image to file
        print('\r\tProgress: 100% \n\t', end='')
        full_filename = self.get_filename()
        self.scene.save_image(full_filename)
