import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm

from src.common.color import RGBColor


def estimate_numerical_error(reference_path, result_path):
    reference = np.load(reference_path)
    result = np.load(result_path)
    shape = reference.shape

    sum_of_errors = 0
    if result.shape != shape:
        raise Exception(f'Image shape {result.shape} does not match with reference shape {shape}')

    for i in range(shape[0]):
        for j in range(shape[1]):
            sum_of_errors += np.linalg.norm(reference[i,j] - result[i,j])
    return sum_of_errors/(shape[0]*shape[1])


def display_error_as_image(reference_path, result_path):
    def save_image(image, full_filename):
        tone_mapper = cv2.createTonemap(gamma=2.5)
        tone_mapped_image = tone_mapper.process(image.astype(np.single))

        plt.imsave(full_filename + '.png', np.clip(tone_mapped_image, 0, 1))
        np.save(full_filename, image)
        cv2.imwrite(full_filename + '.hdr', cv2.cvtColor(image.astype('float32'), cv2.COLOR_RGB2BGR))
        print("Error image saved")

    def show_image(image):
        #tone_mapper = cv2.createTonemap(gamma=2.5)
        #tone_mapped_image = tone_mapper.process(image.astype(np.single))
        #print(tone_mapped_image)
        cv2.imshow('Error', cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)

    reference = np.load(reference_path)
    result = np.load(result_path)
    shape = reference.shape
    error_image = np.zeros([shape[0], shape[1], 3])
    if result.shape != shape:
        raise Exception(f'Image shape {result.shape} does not match with reference shape {shape}')

    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    max_error=0
    min_error=100
    cmap = cm.plasma
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    for i in range(shape[0]):
        for j in range(shape[1]):
            error = np.linalg.norm(reference[i, j] - result[i, j])
            if error>max_error:
                max_error=error
            if error<min_error:
                min_error=error
            cm_val = cmap(norm(error))
            error_image[i, j, 0] = cm_val[0]
            error_image[i, j, 1] = cm_val[1]
            error_image[i, j, 2] = cm_val[2]
    print(min_error)
    print(max_error)
    save_image(error_image, 'src/out/error_estimate')
    show_image(error_image)


if __name__ == "__main__":
    reference_image_path = 'src/AppRenderer/image_error/reference_image/rendered_image_MC_2000_samples.npy'
    image_path = 'src/out/rendered_image_MC_5_samples.npy'
    print(estimate_numerical_error(reference_image_path, image_path))
    display_error_as_image(reference_image_path, image_path)
