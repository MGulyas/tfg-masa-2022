from abc import ABC, abstractmethod  # Abstract Base Class

# Base class for pdfs oer the hemisphere 2*pi
class PDF(ABC):

    @abstractmethod
    def get_val(self, omega_i):
        pass

    @abstractmethod
    def generate_dir(self, u1, u2):
        pass