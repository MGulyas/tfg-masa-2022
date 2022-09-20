from src.common.color import RED, BLACK
from src.integrators.integrator import Integrator


class IntersectionIntegrator(Integrator):

    def __init__(self, filename_):
        super().__init__(filename_ + '_Intersection')

    def compute_color(self, ray, scene):
        if scene.any_hit(ray):
            return RED
        return BLACK