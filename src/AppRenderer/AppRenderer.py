import time

# --------------------------------------------------Set up variables
import cv2
import numpy as np

from src.AppRenderer.image_error.error_estimator import estimate_numerical_error, display_error_as_image
from src.AppRenderer.integrators.bayesian_monte_carlo_integrator import BayesianMonteCarloIntegrator
from src.AppRenderer.integrators.classical_monte_carlo_integrator import ClassicalMonteCarloIntegrator
from src.AppRenderer.integrators.improved_bayesian_monte_carlo_integrator import ImprovedBayesianMonteCarloIntegrator
from src.AppRenderer.integrators.with_prior_bayesian_monte_carlo_integrator import WithPriorBayesianMonteCarloIntegrator
from src.AppRenderer.scenes.classical_monte_carlo_scene import classical_monte_carlo_scene
from src.AppRenderer.scenes.sphere_test import sphere_test_scene
from src.common.gaussian_process.covariance_functions.sobolev import Sobolev
from src.common.gaussian_process.gaussian_process import GP
from src.common.hemisphere_functions.constant import Constant

FILENAME = 'rendered_image'
# DIRECTORY = '.\\out\\'
DIRECTORY = './src/out/'

# -------------------------------------------------Main
# Create Integrator
n = 2
#gaussian_process = GP(cov_func=Sobolev(), p_func=Constant(1))
#integrator = BayesianMonteCarloIntegrator(n, gaussian_process, DIRECTORY + FILENAME)
#integrator = ClassicalMonteCarloIntegrator(n, DIRECTORY + FILENAME)
integrator = ImprovedBayesianMonteCarloIntegrator(n, DIRECTORY + FILENAME)
#integrator = WithPriorBayesianMonteCarloIntegrator(n, DIRECTORY + FILENAME)


# Create the scene #TODO create switch with scene name input to choose from scenes
scene = classical_monte_carlo_scene(use_env_map=True)
#scene = cornell_box_scene(0.75, 2, areaLS=False)
#scene = sphere_test_scene(areaLS=True)

# Attach the scene to the integrator
integrator.add_scene(scene)

# Render!
start_time = time.time()
integrator.prerender()
integrator.render()
end_time = time.time() - start_time
print("--- Rendering time: %s seconds ---" % end_time)

# -------------------------------------------------open saved npy image
print(integrator.get_filename())
image_nd_array = np.load(integrator.get_filename() + '.npy')
tonemapper = cv2.createTonemap(gamma=2.5)

image_nd_array_ldr = tonemapper.process(image_nd_array.astype(np.single)) * 255.0

cv2.imshow('Ray Tracer', cv2.cvtColor(image_nd_array_ldr.astype(np.uint8), cv2.COLOR_BGR2RGB))
cv2.waitKey(0)
