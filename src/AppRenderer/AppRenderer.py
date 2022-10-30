import time

# --------------------------------------------------Set up variables
import cv2
import numpy as np

from src.AppRenderer.integrators.classical_monte_carlo_integrator import ClassicalMonteCarloIntegrator
from src.AppRenderer.scenes.classical_monte_carlo_scene import classical_monte_carlo_scene

FILENAME = 'rendered_image'
# DIRECTORY = '.\\out\\'
DIRECTORY = './out/'

# -------------------------------------------------Main
# Create Integrator
n=20
integrator = ClassicalMonteCarloIntegrator(n, DIRECTORY + FILENAME)

# Create the scene #TODO create switch with scene name input to choose from scenes
scene = classical_monte_carlo_scene(use_env_map=True)
#scene = cornell_box_scene(0.75, 2, areaLS=False)

# Attach the scene to the integrator
integrator.add_scene(scene)

# Render!
start_time = time.time()
integrator.render()
end_time = time.time() - start_time
print("--- Rendering time: %s seconds ---" % end_time)

# -------------------------------------------------open saved npy image
image_nd_array = np.load(integrator.get_filename() + '.npy')
tonemapper = cv2.createTonemap(gamma=2.5)
image_nd_array_ldr = tonemapper.process(image_nd_array.astype(np.single)) * 255.0
cv2.imshow('Ray Tracer MLCG 2021-2022', cv2.cvtColor(image_nd_array_ldr.astype(np.uint8), cv2.COLOR_BGR2RGB))
cv2.waitKey(0)
