from src.common import RGBColor, BLACK
from src.common import Vector3D
from src.AppRenderer.core import Camera
from src.AppRenderer.core import PointLight
from src.AppRenderer.core import Lambertian
from src.AppRenderer.core import Parallelogram
from src.AppRenderer.core import Sphere
from src.AppRenderer.core import Scene


def cornell_box_scene(dist, side, areaLS=False):
    # Create a scene object
    scene_ = Scene()
    i_a = RGBColor(1.0, 1.0, 0.3)
    scene_.set_ambient(i_a)

    # some useful values to create the box
    z_far = -(dist + side)
    x_left = -side / 2
    x_right = side / 2
    y_bottom = -side / 2
    y_top = side / 2
    plane_point_ll = Vector3D(x_left, y_bottom, z_far)  # lower left corner
    plane_point_ur = Vector3D(x_right, y_top, z_far)  # upper right corner

    # Create the materials (BRDF)
    floor_material = Lambertian(RGBColor(0.8, 0.8, 0.8))
    red_material = Lambertian(RGBColor(0.7, 0.2, 0.2))
    green_material = Lambertian(RGBColor(0.2, 0.7, 0.2))
    blue_material = Lambertian(RGBColor(0.2, 0.2, 0.7))
    black_material = Lambertian(BLACK)

    # Create the Scene Geometry (3D objects)
    # sphere
    sphere = Sphere(Vector3D(0.0, 0.0, -(dist + side / 2.0)), 0.25)
    sphere.set_BRDF(blue_material)
    scene_.add_object(sphere)
    # back wall
    right_vector = Vector3D(side, 0.0, 0.0)
    up_vector = Vector3D(0.0, side, 0.0)
    plane = Parallelogram(plane_point_ll, right_vector, up_vector)
    plane.set_BRDF(floor_material)
    scene_.add_object(plane)
    # floor
    back_vector = Vector3D(0.0, 0.0, side)
    plane = Parallelogram(plane_point_ll, right_vector, back_vector)
    plane.set_BRDF(floor_material)
    scene_.add_object(plane)
    # left wall
    plane = Parallelogram(plane_point_ll, up_vector, back_vector)
    plane.set_BRDF(red_material)
    scene_.add_object(plane)
    # ceiling
    left_vector = Vector3D(-side, 0.0, 0.0)
    plane = Parallelogram(plane_point_ur, left_vector, back_vector)
    plane.set_BRDF(floor_material)
    scene_.add_object(plane)
    # right wall
    bottom_vector = Vector3D(0.0, -side, 0.0)
    plane = Parallelogram(plane_point_ur, bottom_vector, back_vector)
    plane.set_BRDF(green_material)
    scene_.add_object(plane)

    if areaLS:
        # For Monte Carlo-based integrators
        # Create are light source
        side_ls = side / 2
        delta_ls = (side - side_ls) / 2
        light_source_point = Vector3D(x_left + delta_ls,
                                      y_top - 0.0001,
                                      -(dist + delta_ls))
        i_l = 1
        plane = Parallelogram(light_source_point,
                              Vector3D(side_ls, 0, 0),
                              Vector3D(0, 0, -side_ls),
                              RGBColor(i_l, i_l, i_l))
        plane.set_BRDF(black_material)
        scene_.add_object(plane)
    else:
        # For Phong Integrator: Create three Point Light Sources
        i_l = 0.5
        delta_y = 0.5
        point_light_1 = PointLight(Vector3D(0.0, side / 2 - delta_y, -(dist + side / 2.0)),
                                   RGBColor(i_l, i_l, i_l))
        scene_.add_point_light_sources(point_light_1)
        point_light_2 = PointLight(Vector3D(-side / 4, side / 2 - delta_y, -(dist + side / 2.0)),
                                   RGBColor(i_l, i_l, i_l))
        scene_.add_point_light_sources(point_light_2)
        point_light_3 = PointLight(Vector3D(side / 4, side / 2 - delta_y, -(dist + side / 2.0)),
                                   RGBColor(i_l, i_l, i_l))
        scene_.add_point_light_sources(point_light_3)

    # Set up an environment map
    env_map_path = '../env_maps/arch_nozero.hdr'
    scene_.set_environment_map(env_map_path)

    # Create the camera (always centred at 0,0,0)
    width = 500
    height = 500
    vertical_fov = 70
    camera = Camera(width, height, vertical_fov)
    scene_.set_camera(camera)

    return scene_
