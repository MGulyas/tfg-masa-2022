from abc import abstractmethod, ABC


class CovarianceFunction(ABC):
    @abstractmethod
    def eval(self, omega_i, omega_j):
        pass