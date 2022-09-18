import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.common.environment_map import EnvironmentMap
from src.common.hit_data import HitData


class Scene:
    def __init__(self):
        self.camera = None
        self.env_map = None  # not initialized
        self.rendered_image = None
        self.object_list = []  # object list
        self.pointLights = []  # list of point light sources (for Phong Illumination)
        self.i_a = None

    def set_ambient(self, i_a):
        self.i_a = i_a

    # set camera
    def set_camera(self, camera):
        self.camera = camera
        self.rendered_image = np.zeros((camera.height, camera.width, 3))

    # set environment map
    def set_environment_map(self, env_map_path):
        self.env_map = EnvironmentMap(env_map_path)

    # add objects
    def add_object(self, new_object):
        self.object_list.append(new_object)

    # add point light sources
    def add_point_light_sources(self, point_light):
        self.pointLights.append(point_light)

    def any_hit(self, ray):
        # ASSIGNMENT 1.2: PUT YOUR CODE HERE
        return False

    def closest_hit(self, ray):
        # find closest hit object, its distance, hit_point and normal
        # scan through primitives in scene, find closest
        hit_data = HitData()
        for i in range(len(self.object_list)):
            this_hit = self.object_list[i].intersect(ray)
            if this_hit.has_hit:  # Hit
                if this_hit.hit_distance < hit_data.hit_distance:  # Distance
                    hit_data = this_hit
                    hit_data.primitive_index = i
        return hit_data

    # save pixel array to file
    def save_image(self, full_filename):
        tonemapper = cv2.createTonemap(gamma=2.5)
        image_nd_array_ldr = tonemapper.process(self.rendered_image.astype(np.single))
        plt.imsave(full_filename + '.png', np.clip(image_nd_array_ldr, 0, 1))
        np.save(full_filename, self.rendered_image)
        cv2.imwrite(full_filename + '.hdr', cv2.cvtColor(self.rendered_image.astype('float32'), cv2.COLOR_RGB2BGR))
        print("Image Saved")

    # set pixel value
    def set_pixel(self, pixel_val, x, y):
        # pixel_val.clamp(0.0, 1.0)
        self.rendered_image[y, x, 0] = pixel_val.r
        self.rendered_image[y, x, 1] = pixel_val.g
        self.rendered_image[y, x, 2] = pixel_val.b

