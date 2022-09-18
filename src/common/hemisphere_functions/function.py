from abc import ABC, abstractmethod


class Function(ABC):
    def __init__(self, integral_value):
        self.ground_truth = integral_value

    @abstractmethod
    def eval(self, omega_i):
        pass

    @abstractmethod
    def get_integral(self):
        pass