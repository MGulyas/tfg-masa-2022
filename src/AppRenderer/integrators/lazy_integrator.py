from src.common import BLACK
from src.AppRenderer.integrators import Integrator


class LazyIntegrator(Integrator):
    def __init__(self, filename_):
        super().__init__(filename_ + '_Intersection')

    def compute_color(self, ray, scene):
        return BLACK
