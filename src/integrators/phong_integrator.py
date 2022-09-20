from src.common.color import BLACK
from src.integrators.integrator import Integrator


class PhongIntegrator(Integrator):

    def __init__(self, filename_):
        super().__init__(filename_ + '_Phong')

    def compute_color(self, ray, scene):
        hit_data = scene.closest_hit(ray)
        if hit_data.has_hit:
            kd = scene.object_list[hit_data.primitive_index].get_BRDF().kd
            ia = scene.i_a
            La = ia.multiply(kd)
            d = hit_data.hit_distance
            return kd
        return BLACK
