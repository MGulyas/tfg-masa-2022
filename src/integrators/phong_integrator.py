import math

from src.common.color import BLACK, RED, WHITE
from src.common.ray import Ray
from src.common.vector_3d import direction_vector, distance, Dot, Normalize, Vector3D
from src.integrators.integrator import Integrator


class PhongIntegrator(Integrator):

    def __init__(self, filename_):
        super().__init__(filename_ + '_Phong')

    def compute_color(self, ray, scene):
        hit_data = scene.closest_hit(ray)
        if hit_data.has_hit:
            color = BLACK
            n = Normalize(hit_data.normal)  # surface normal
            for light in scene.pointLights:
                L = direction_vector(hit_data.hit_point, light.pos)  # point to light direction
                I = self._incident_intensity(scene, hit_data, light, L)
                color += self._diffuse_reflection(scene, hit_data, n, L, I)
                #color += self._specular_reflection(hit_data, ray, n, L, I)
            return color + self._ambient_reflection(scene, hit_data)
        return BLACK

    def _incident_intensity(self, scene, hit_data, light, L):
        light_to_hit_point_distance = distance(light.pos, hit_data.hit_point)
        shadow_ray = Ray(origin=hit_data.hit_point, direction=L, tmax=light_to_hit_point_distance)
        if scene.any_hit(shadow_ray):
            I = BLACK
        else:
            I = light.intensity / light_to_hit_point_distance ** 2  # incident intensity from light source on point
        return I

    def _diffuse_reflection(self, scene, hit_data, n, L, I):
        kd = scene.object_list[hit_data.primitive_index].get_BRDF().kd
        return kd.multiply(I * max(0, Dot(n, L)))


    def _specular_reflection(self, hit_data, ray, n, L, I):
        ks = WHITE
        s = 3
        # ks = scene.object_list[hit_data.primitive_index].get_BRDF().ks
        # s = scene.object_list[hit_data.primitive_index].get_BRDF().s
        r = Normalize(n * 2 * Dot(n, L) - L)
        v = direction_vector(hit_data.hit_point, ray.o)  # viewer direction
        return ks.multiply(I * max(0, Dot(r, v) ** s))

    def _ambient_reflection(self, scene, hit_data):
        ka = scene.object_list[hit_data.primitive_index].get_BRDF().kd
        return ka.multiply(scene.i_a)