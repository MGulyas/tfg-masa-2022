import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# function to display the difference pixel by pixel between two images of the same sizes given their paths
# the difference is displayed as a color image where the color is proportional to the difference.
# the color of the output image is a gradient from red to green where red means the difference
# is high and green means the difference is low.  The output image is saved in the path given by parameter
from matplotlib import cm


def image_difference(image1, image2, output, normalize=False, colormap=None):
    # load the images
    image1 = np.load(image1)
    image2 = np.load(image2)

    # check if the images have the same size
    if image1.shape != image2.shape:
        raise Exception(f'Image shape {image1.shape} does not match with reference shape {image2.shape}')

    # initialize the output image
    shape = image1.shape
    error_image = np.zeros([shape[0], shape[1], 3])

    # compute the difference between the two images
    errors = []
    for i in range(shape[0]):
        errors_j = []
        for j in range(shape[1]):
            errors_j.append(np.linalg.norm(image1[i, j] - image2[i, j]))
        errors.append(errors_j[:])
    if colormap and normalize:
        max_error = np.max(errors)
        min_error = np.min(errors)
        norm = matplotlib.colors.Normalize(vmin=min_error, vmax=max_error)
    for i in range(shape[0]):
        for j in range(shape[1]):
            error = errors[i][j]
            if colormap is None:
                error_image[i, j] = [error, error, error]
            else:
                if normalize:
                    error_image[i, j] = colormap(norm(error))[:3]
                else:
                    error_image[i, j] = colormap(error)[:3]
    if normalize and colormap is None:
        error_image = (error_image - np.min(error_image)) / np.max(error_image)
    print(np.min(errors))
    print(np.max(errors))
    print(np.min(error_image))
    print(np.max(error_image))

    # save the error image
    plt.imsave(output + '.png', error_image)
    np.save(output, error_image)
    print("Error image saved")

if __name__ == "__main__":
    #image_difference('src/out/rendered_image_improved_BMC_with_prior_30_samples_4gps.npy', 'src/AppRenderer/image_error/reference_image/rendered_image_MC_3000_samples.npy', 'src/out/error_estimatee.npy')
    image_difference('src/out/rendered_image_MC_30_samples.npy', 'src/AppRenderer/image_error/reference_image/rendered_image_MC_3000_samples.npy', 'src/out/error_estimate_monte_carlo_normalized_without_colormap.npy', normalize=True, colormap=cm.rainbow)