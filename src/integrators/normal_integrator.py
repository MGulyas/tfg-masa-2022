from src.common.color import RGBColor, BLACK
from src.common.vector_3d import Vector3D
from src.integrators.integrator import Integrator


class NormalIntegrator(Integrator):

    def __init__(self, filename_):
        super().__init__(filename_ + '_Normal')

    def compute_color(self, ray, scene):
        hit_data = scene.closest_hit(ray)
        if hit_data.has_hit:
            color = (hit_data.normal + Vector3D(1, 1, 1))/2
            return RGBColor(color.x, color.y, color.z)
        return BLACK