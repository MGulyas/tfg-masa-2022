from math import acos, atan2, floor
import cv2

from src.common.color import RGBColor
from src.common.constants import PI


class EnvironmentMap:
    def __init__(self, env_map_path):
        # IMREAD_ANYDEPTH is needed because even though the data is stored in 8-bit channels
        # when it's read into memory it's represented at a higher bit depth
        self.env_map_hdr = cv2.imread(env_map_path, flags=cv2.IMREAD_ANYDEPTH)
        self.env_map_hdr = cv2.cvtColor(self.env_map_hdr, cv2.COLOR_RGB2BGR)
        self.height = self.env_map_hdr.shape[0]
        self.width = self.env_map_hdr.shape[1]

    def euclideanToLatLong(self, d):
        uLatLong = (1 + (1 / PI) * atan2(d.x, -d.z)) / 2.0
        vLatLong = (1 / PI) * acos(d.y)
        return (uLatLong, vLatLong)

    def getValue(self, d):
        (u, v) = self.euclideanToLatLong(d)
        tx = floor(u * (self.width - 1))  # texel x coordinate
        ty = floor(v * (self.height - 1))  # texel y coordinate
        res = self.env_map_hdr[ty, tx, :]
        return RGBColor(res[0], res[1], res[2])