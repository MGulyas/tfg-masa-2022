from src.common.color import BLACK
from src.integrators.integrator import Integrator


class LazyIntegrator(Integrator):
    def __init__(self, filename_):
        super().__init__(filename_ + '_Intersection')

    def compute_color(self, ray, scene):
        return BLACK
