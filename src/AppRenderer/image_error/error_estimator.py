import cv2
import numpy as np
import matplotlib.pyplot as plt


def estimate_numerical_error(reference_path, result_path):
    reference = np.load(reference_path)
    result = np.load(result_path)
    size = reference.size
    sum_of_errors = 0
    if result.size != size:
        raise Exception(f'Image size {result.size} does not match with reference size {size}')
    for x, y in enumerate(size):
        sum_of_errors += np.linalg.norm(reference[x,y], result[x,y])
    return sum_of_errors/(size[0]*size[1])


def display_error_as_image(reference_path, result_path):
    def save_image(image, full_filename):
        tone_mapper = cv2.createTonemap(gamma=2.5)
        tone_mapped_image = tone_mapper.process(image.astype(np.single))

        plt.imsave(full_filename + '.png', np.clip(tone_mapped_image, 0, 1))
        np.save(full_filename, image)
        cv2.imwrite(full_filename + '.hdr', cv2.cvtColor(image.astype('float32'), cv2.COLOR_RGB2BGR))
        print("Error image saved")

    def show_image(image):
        tone_mapper = cv2.createTonemap(gamma=2.5)
        tone_mapped_image = tone_mapper.process(image.astype(np.single))
        cv2.imshow('Error', cv2.cvtColor(tone_mapped_image.astype(np.uint8), cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)

    reference = np.load(reference_path)
    result = np.load(result_path)
    size = reference.size
    error = np.zeros(size[0], size[1], 3) #image will be b/w. could do it in red maybe
    if result.size != size:
        raise Exception(f'Image size {result.size} does not match with reference size {size}')
    for x, y in enumerate(size):
        error[x,y, 0] = np.linalg.norm(reference[x,y], result[x,y])
    save_image(error, './out/error_estimate')
    show_image(error)
