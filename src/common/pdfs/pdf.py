from abc import ABC, abstractmethod  # Abstract Base Class

# Base class for pdfs oer the hemisphere 2*pi
class PDF(ABC):

    @abstractmethod
    # returns the probability of generating the direction omega_i
    def get_val(self, omega_i):
        pass

    @abstractmethod
    # given two random numbers u1, u2 uniformly distributed in 0,1 returns a spherical direction
    # TODO why not having generate_dir return both the direction and the probabilty so only one function has to be called externally?
    def generate_dir(self, u1, u2):
        pass