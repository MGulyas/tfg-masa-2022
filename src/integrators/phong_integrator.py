from src.integrators.integrator import Integrator


class PhongIntegrator(Integrator):

    def __init__(self, filename_):
        super().__init__(filename_ + '_Phong')

    def compute_color(self, ray, scene):
        # ASSIGNMENT 1.4: PUT YOUR CODE HERE
        pass