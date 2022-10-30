from math import sqrt

import numpy as np

from src.common.color import BLACK
from src.AppRenderer.core.hit_data import HitData
from src.common.vector_3d import Normalize, Dot
from src.AppRenderer.core.object.primitive import Primitive


class Sphere(Primitive):
    # Initializer
    def __init__(self, sphere_origin, sphere_radius, emission=BLACK):
        super().__init__(emission)
        self.origin = sphere_origin
        self.radius = sphere_radius
        self.radius_squared = sphere_radius * sphere_radius  # optimization

    # Member Functions
    # Returns tuple of (bool hit, distance, hit_point, normal)
    def intersect(self, ray):
        ray_dir = Normalize(ray.d)
        temp = np.subtract(ray.o, self.origin)
        A = Dot(ray_dir, ray_dir)
        B = 2.0 * Dot(ray_dir, temp)
        C = Dot(temp, temp) - self.radius_squared

        disc = (B * B) - (4.0 * A * C)  # Discriminant

        if disc < 0.0:  # No Hit
            return HitData()  # return an 'empty' HitData object (no intersection)

        sqrt_disc = sqrt(disc)  # square root of discriminant

        t_small = (-B - sqrt_disc) / (2.0 * A)
        if t_small >= ray.t_min and t_small <= ray.t_max:  # Hit
            p = ray.get_hitpoint(t_small)
            n = Normalize((p - self.origin) / self.radius)
            return HitData(has_hit=True, hit_point=p, normal=n, hit_distance=t_small)

        t_large = (-B + sqrt_disc) / (2.0 * A)
        if t_large >= ray.t_min and t_large <= ray.t_max:  # Hit
            p = ray.get_hitpoint(t_large)
            n = Normalize((p - self.origin) / self.radius)
            return HitData(has_hit=True, hit_point=p, normal=n, hit_distance=t_large)

        # Ray did not intersect sphere
        return HitData()
