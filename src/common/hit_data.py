import numpy as np
from src.common.constants import HUGEVALUE


class HitData:
    def __init__(self, has_hit=False, hit_point=np.zeros(3), normal=np.zeros(3),
                 hit_distance=HUGEVALUE, primitive_index=-1):
        self.has_hit = has_hit  # whether or not this object represents a hit
        self.hit_point = hit_point  # hit point
        self.normal = normal  # normal at the surface
        self.hit_distance = hit_distance  # intersection distance along the ray
        self.primitive_index = primitive_index  # index of the object (primitive) hit by the ray
