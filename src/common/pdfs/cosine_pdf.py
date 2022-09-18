from math import sqrt, cos, sin, pi

# PDF: p(omega) = (n+1)/(2*pi) * cos(theta)**n
from src.common.constants import TWO_PI
from src.common.pdfs.pdf import PDF
from src.common.vector_3d import Vector3D, Dot


class CosinePDF(PDF):
    def __init__(self, exp_):
        self.exp = exp_

    def get_val(self, omega_i):
        normal = Vector3D(0.0, 1.0, 0.0)
        cos_theta = Dot(normal, omega_i)
        return (self.exp + 1) / (2 * pi) * cos_theta ** self.exp

    def generate_dir(self, u1, u2):
        # compute the y coordinate (up)
        y = pow(u2, 1.0 / (self.exp + 1.0))
        # compute the x and z coordinates
        phi = TWO_PI * u1  # azimuth angle (aka rotation angle)
        aux_power = 2.0 / (self.exp + 1.0)
        aux_sqrt = sqrt(1 - u2 ** aux_power)
        x = sin(phi) * aux_sqrt
        z = cos(phi) * aux_sqrt
        return Vector3D(x, y, z)