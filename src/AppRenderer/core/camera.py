from math import tan

from src.common.constants import PI
from src.common.vector_3d import Vector3D, Normalize


class Camera:
    def __init__(self, width_, height_, vertical_fov_):
        self.width = width_
        self.height = height_
        self.vertical_fov = vertical_fov_ / 180 * PI
        self.aspect_ratio = width_ / height_

    def get_direction(self, x, y):
        # Convert from pixel coordinates to screen space
        x_ss = 2.0 * (x + 0.5) / self.width - 1.0
        y_ss = 1.0 - 2.0 * (y + 0.5) / self.height
        # Convert from screen space to camera space
        tan_half_fov = tan(self.vertical_fov / 2.0)
        x_cs = x_ss * tan_half_fov * self.aspect_ratio
        y_cs = y_ss * tan_half_fov
        p_cs = Vector3D(x_cs, y_cs, -1.0)
        # Compute the ray direction in camera space
        direction = Normalize(p_cs)  # because camera is always at (0,0,0)
        return direction
