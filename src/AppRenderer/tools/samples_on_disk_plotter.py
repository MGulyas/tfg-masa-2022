from cmath import acos, pi
from math import atan2

import numpy as np
from matplotlib import pyplot as plt

from src.common.vector_3d import Vector3D, Dot


def euclidean_to_disk(sample_set):
    ns = len(sample_set)
    r = np.zeros(ns)
    phi = np.zeros(ns)
    normal = Vector3D(0, 1, 0)
    for i in range(ns):
        current_dir = sample_set[i]
        r_i = acos(Dot(normal, current_dir))
        phi_i = atan2(current_dir.x, current_dir.z)
        r[i] = r_i
        phi[i] = phi_i
    return r, phi


# Visualize the samples on a disk

def visualize_sample_set(sample_set, weights=[]):
    r, phi = euclidean_to_disk(sample_set)
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    if len(weights) == 0:
        ax.scatter(phi, r)
    else:
        av_weight_val = 1.0 / len(weights)
        delta = 0.3 * av_weight_val
        cm = plt.get_cmap('jet')
        sc = plt.scatter(phi, r, vmin=av_weight_val - delta, vmax=av_weight_val + delta, c=weights, cmap=cm)
        fig.colorbar(sc)
        ax = plt.gca()
    ax.axis([0, 2 * pi, 0, pi / 2])
    plt.show()