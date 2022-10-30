from src.common.color import BLACK
from src.AppRenderer.core.hit_data import HitData
from src.common.vector_3d import Normalize, Length, Cross, Dot
from src.AppRenderer.core.object.primitive import Primitive

class Parallelogram(Primitive):
    # Initializer
    def __init__(self, point, s1, s2, emission=BLACK):
        super().__init__(emission)
        self.point = point  # point (a corner of the rectangle
        self.s1 = s1  # side 1
        self.s2 = s2  # side 2
        self.s1_n = Normalize(s1)
        self.s2_n = Normalize(s2)
        self.s1_l = Length(s1)
        self.s2_l = Length(s2)
        self.normal = Normalize(Cross(s1, s2))

    # Member Functions
    # Returns tuple of (bool hit, distance, hit_point, normal)
    def intersect(self, ray):
        ray_dir = Normalize(ray.d)
        normal_ = self.normal
        denominator = Dot(ray_dir, normal_)
        if denominator == 0.0:  # Check for division by zero
            # ray is parallel, no hit
            return HitData()

        t = Dot(normal_, (self.point - ray.o)) / denominator
        if t >= ray.t_min and t <= ray.t_max:  # Hit
            p_hit = ray.get_hitpoint(t)
            # Check whether p is within the square limits
            p_ph = p_hit - self.point  # 3D vector from point to p_hit

            # Project p_ph onto s1 and s2
            p_ph_n = Normalize(p_ph)
            p_ph_l = Length(p_ph)
            cos_alpha1 = Dot(self.s1_n, p_ph_n)
            cos_alpha2 = Dot(self.s2_n, p_ph_n)
            q1 = cos_alpha1 * p_ph_l
            q2 = cos_alpha2 * p_ph_l

            if q1 < 0.0 or q2 < 0.0 or q1 > self.s1_l or q2 > self.s2_l:
                return HitData()

            if Dot(self.normal, ray_dir) > 0:
                normal_ = self.normal * (-1)
            return HitData(has_hit=True, hit_point=p_hit, normal=normal_, hit_distance=t)

        # Ray did not intersect plane
        return HitData()
