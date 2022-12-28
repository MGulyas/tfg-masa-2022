# SETUP
# name of the used methods, and their markers (for plotting)
from src.common.hemisphere_functions.constant import Constant
from src.common.hemisphere_functions.cosine_lobe import CosineLobe
from src.common.pdfs.cosine_pdf import CosinePDF
from src.common.pdfs.uniform_pdf import UniformPDF
import numpy as np


def set_up_ns_vector():
    ns_min = 20  # minimum number of samples (ns) used for the Monte Carlo estimate
    ns_max = 61  # maximum number of samples (ns) used for the Monte Carlo estimate
    ns_step = 20  # step for the number of samples
    return np.arange(start=ns_min, stop=ns_max, step=ns_step)  # the number of samples to use per estimate


# methods_label = [('MC', 'o'), ('BMC', 'x'), ('MC IS', 'v'), ('BMC IS', '1'), ('BMC WITH PRIOR', '2')]
methods_label = [('MC', 'o'), ('BMC', 'x'), ('BMC WITH PRIOR', '2')]
n_methods = len(methods_label)  # number of tested monte carlo methods

# Set up the function we wish to integrate
# We will consider integrals of the form: L_i * brdf * cos
l_i = Constant(5)
brdf = Constant(1)
cosine_term = CosineLobe(1)

integrand = [l_i, brdf, cosine_term]  # l_i * brdf * cos

# Set-up the pdf used to sample the hemisphere
pdf = UniformPDF()

#importance_sampling_pdf = CosinePDF(1)

# Compute/set the ground truth value of the integral we want to estimate
# NOTE: in practice, when computing an image, this value is unknown
ground_truth = 5*cosine_term.get_integral()  # Assuming that L_i = 5 and BRDF = 1
print('Ground truth: ' + str(ground_truth))

# Experimental set-up

ns_vector = set_up_ns_vector()
n_estimates = 5  # the number of estimates to perform for each value in ns_vector

n_runs = 3
