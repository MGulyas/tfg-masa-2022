from abc import ABC, abstractmethod


class BRDF(ABC):
    @abstractmethod
    def get_value(self, incoming, outgoing, normal):
        pass