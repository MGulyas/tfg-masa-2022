# Lambertian (perfect diffuse material)
from src.common.color import BLACK
from src.common.constants import INVERTED_PI
from src.common.vector_3d import Dot
from src.core.brdf import BRDF


class Lambertian(BRDF):
    # Initializer
    def __init__(self, diffuse_colour):
        self.kd = diffuse_colour * INVERTED_PI

    # Member Functions
    def get_value(self, wi, wo, normal):
        cos_n_wi = Dot(normal, wi)
        if Dot(normal, wi) > 0.0:
            return self.kd * cos_n_wi  # Colour
        else:
            return BLACK