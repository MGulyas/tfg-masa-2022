from src.integrators.integrator import Integrator


class DepthIntegrator(Integrator):

    def __init__(self, filename_, max_depth_=10):
        super().__init__(filename_ + '_Depth')
        self.max_depth = max_depth_

    def compute_color(self, ray):
        # ASSIGNMENT 1.3: PUT YOUR CODE HERE
        pass
