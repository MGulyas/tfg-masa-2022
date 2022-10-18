# SETUP
# name of the used methods, and their markers (for plotting)
from src.common.hemisphere_functions.constant import Constant
from src.common.hemisphere_functions.cosine_lobe import CosineLobe
from src.common.pdfs.uniform_pdf import UniformPDF
import numpy as np

def set_up_ns_vector():
    ns_min = 20  # minimum number of samples (ns) used for the Monte Carlo estimate
    ns_max = 1001  # maximum number of samples (ns) used for the Monte Carlo estimate
    ns_step = 20  # step for the number of samples
    return np.arange(start=ns_min, stop=ns_max, step=ns_step)  # the number of samples to use per estimate

methods_label = [('MC', 'o')]
# methods_label = [('MC', 'o'), ('MC IS', 'v'), ('BMC', 'x'), ('BMC IS', '1')]
n_methods = len(methods_label) # number of tested monte carlo methods

# Set up the function we wish to integrate
# We will consider integrals of the form: L_i * brdf * cos
l_i = Constant(1)
kd = 1
brdf = Constant(kd)
cosine_term = CosineLobe(1)
integrand = [l_i, brdf, cosine_term]  # l_i * brdf * cos

# Set-up the pdf used to sample the hemisphere
pdf = UniformPDF()

# Compute/set the ground truth value of the integral we want to estimate
# NOTE: in practice, when computing an image, this value is unknown
ground_truth = cosine_term.get_integral()  # Assuming that L_i = 1 and BRDF = 1
print('Ground truth: ' + str(ground_truth))

# Experimental set-up

ns_vector = set_up_ns_vector()
n_estimates = 1  # the number of estimates to perform for each value in ns_vector


# Initialize a matrix of estimate error at zero
results = np.zeros((len(ns_vector), n_methods))  # Matrix of average error