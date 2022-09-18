from src.common.color import BLACK
from src.common.hit_data import HitData
from src.common.vector_3d import Normalize, Dot
from src.core.object.primitive import Primitive


class InfinitePlane(Primitive):
    # Initializer
    def __init__(self, plane_origin, plane_normal, emission=BLACK):
        super().__init__(emission)
        self.origin = plane_origin
        self.normal = Normalize(plane_normal)

    # Member Functions
    # Returns tuple of (bool hit, distance, hit_point, normal)
    def intersect(self, ray):
        ray_dir = Normalize(ray.d)
        denominator = Dot(ray_dir, self.normal)
        if denominator == 0.0:  # Check for division by zero
            # ray is parallel, no hit
            return HitData()

        t = Dot(self.normal, (self.origin - ray.o)) / denominator
        if t >= ray.t_min and t <= ray.t_max:  # Hit
            p = ray.get_hitpoint(t)
            return HitData(has_hit=True, hit_point=p, normal=self.normal, hit_distance=t)

        # Ray did not intersect plane
        return HitData()