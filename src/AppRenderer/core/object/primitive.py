from abc import ABC, abstractmethod

from src.common.color import BLACK


class Primitive(ABC):
    def __init__(self, emission=BLACK):
        self.emission = emission
        self.BRDF = None

    @abstractmethod
    def intersect(self, ray):
        pass

    # Setters
    def set_BRDF(self, BRDF):
        self.BRDF = BRDF

    # Getters
    def get_BRDF(self):
        return self.BRDF
