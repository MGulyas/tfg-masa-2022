from math import acos, cos, sin, atan2,  pi
from random import random
import numpy as np
import matplotlib.pyplot as plt

# Convert a point set defined on the unit sphere from euclidean coordinates (3D) to 2D polar coordinates (disk)
# Returns two np arrays
from src.common.vector_3d import *


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


def sample_set_hemisphere(n_samples, pdf):
    sample_set = []
    sample_prob = []
    for i in range(n_samples):
        u1 = random()
        u2 = random()
        omega_i = pdf.generate_dir(u1, u2)
        sample_set.append(omega_i)
        sample_prob.append(pdf.get_val(omega_i))
        # plt.plot(omega_i.x, omega_i.z, 'o')
    # plt.show()
    return sample_set, sample_prob


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


def oriented_hemi_dir(pdf, u1, u2, new_normal):
    dir_ = pdf.generate_dir(u1, u2)  # random point on hemisphere
    return center_around_normal(dir_, new_normal)  # normalized


def rotate_around_y(alpha, dir_):
    sin_alpha = sin(alpha)
    cos_alpha = cos(alpha)
    new_x = dir_.x * cos_alpha + dir_.z * sin_alpha
    new_z = -dir_.x * sin_alpha + dir_.z * cos_alpha
    return Vector3D(new_x, dir_.y, new_z)


def center_around_normal(dir, normal):
    # create orthonormal basis around normal
    w = normal
    v = Cross(Vector3D(0.00319, 1.0, 0.0078), w)  # jittered up
    v = Normalize(v)  # normalize
    u = Cross(v, w)

    hemi_dir = (v * dir.x) + (w * dir.y) + (u * dir.z)  # project the original direction dir onto the new frame
    return Normalize(hemi_dir)
