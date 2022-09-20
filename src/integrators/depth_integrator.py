from src.common.color import BLACK, RGBColor
from src.integrators.integrator import Integrator


class DepthIntegrator(Integrator):

    def __init__(self, filename_, max_depth_=5):
        super().__init__(filename_ + '_Depth')
        self.max_depth = max_depth_

    def compute_color(self, ray, scene):
        hit_data = scene.closest_hit(ray)
        if hit_data.has_hit:
            c = max(1-hit_data.hit_distance/self.max_depth, 0)
            return RGBColor(c, c, c)
        return BLACK
