from src.common.color import RGBColor
from src.common.vector_3d import Vector3D
from src.AppRenderer.core.camera import Camera
from src.AppRenderer.core.material.lambertian import Lambertian
from src.AppRenderer.core.object.parallelogram import Parallelogram
from src.AppRenderer.core.object.sphere import Sphere
from src.AppRenderer.core.scene import Scene


def classical_monte_carlo_scene(use_env_map=False):
    # Create a scene object
    scene_ = Scene()
    i_a = RGBColor(0.5, 0.5, 0.5)
    #scene_.set_ambient(i_a)

    # Create the materials (BRDF)
    white_diffuse = Lambertian(RGBColor(0.8, 0.8, 0.8))
    pink_diffuse = Lambertian(RGBColor(1, 0.2, 0.5))
    green_diffuse = Lambertian(RGBColor(0.2, 0.8, 0.2))

    # Create the Scene Geometry (3D objects)
    # sphere
    radius = 2
    sphere = Sphere(Vector3D(0.0, 0.0, -5.0), radius)
    sphere.set_BRDF(pink_diffuse)
    scene_.add_object(sphere)


    # Finite plane
    side = 4 * radius
    half_side = side / 2
    plane_point = Vector3D(-half_side, -radius, -5.0 + half_side)
    right_vector = Vector3D(side, 0.0, 0.0)
    front_vector = Vector3D(0.0, 0.0, -side)
    plane = Parallelogram(plane_point, right_vector, front_vector, emission=RGBColor(0,0,0))
    plane.set_BRDF(green_diffuse)
    scene_.add_object(plane)

    #create an area light source
    black_material = Lambertian(RGBColor(1,0,0))
    light_source_point = Vector3D(-half_side, 3 * radius, -5.0 + half_side)
    i_l = 0.5
    area_light_source = Parallelogram(light_source_point,
                                      right_vector,
                                      front_vector,
                                      RGBColor(i_l, i_l, i_l))
    area_light_source.set_BRDF(black_material)
    #scene_.add_object(area_light_source)


    if use_env_map:
        # Set up an environment map
        # env_map_path = 'env_maps/black_and_white.hdr'
        # env_map_path = 'env_maps/outdoor_umbrellas_4k.hdr'
        # env_map_path = 'env_maps/outdoor_umbrellas_4k_clamped.hdr'
        env_map_path = '../src/AppRenderer/env_maps/arch_nozero.hdr'
        scene_.set_environment_map(env_map_path)

    # Create the camera
    width = 500 #500
    height = 500
    vertical_fov = 60
    camera = Camera(width, height, vertical_fov)
    scene_.set_camera(camera)

    return scene_
