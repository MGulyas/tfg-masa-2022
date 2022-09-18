from math import sqrt, cos, sin, pi

# Uniform PDF over the hemisphere: p(omega) = 1/(2*pi)
from src.common.constants import TWO_PI
from src.common.pdfs.pdf import PDF
from src.common.vector_3d import Vector3D


class UniformPDF(PDF):

    def get_val(self, omega_i):
        return 1 / (2 * pi)

    def generate_dir(self, u1, u2):
        # compute the y coordinate (up)
        y = u2
        # compute the x and z coordinates
        phi = TWO_PI * u1  # azimuth angle (aka rotation angle)
        aux_sqrt = sqrt(1 - u2 ** 2)
        x = sin(phi) * aux_sqrt
        z = cos(phi) * aux_sqrt
        return Vector3D(x, y, z)