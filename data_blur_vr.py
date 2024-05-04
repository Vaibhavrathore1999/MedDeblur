import cv2
import numpy as np
import random
from PIL import Image

def load_image(file_path):
    # Load an image using OpenCV
    image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    
    # Convert 64F to 8U if necessary
    if image.dtype == np.float64:
        # Scale to the range 0-255 and convert to uint8
        image = np.uint8(255 * (image - image.min()) / (image.max() - image.min()))
    
    # Check if the image is grayscale and convert to RGB if necessary
    if len(image.shape) == 2:  # Image has only height and width, which means it's grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def generate_random_degree_and_angle(dmin, dmax, amin, amax):
    # Generate random degree and angle
    degree = random.randint(dmin, dmax)
    angle = random.randint(amin, amax)
    return degree, angle

def create_blur_kernel(degree, angle, scale=1.0):
    # Compute rotation matrix and convert to 3x3 by adding a row for affine transformations
    rotation_matrix = cv2.getRotationMatrix2D((degree/2, degree/2), angle, scale)
    rotation_matrix = np.vstack([rotation_matrix, [0, 0, 1]])  # Convert 2x3 to 3x3
    
    # Compute a diagonal matrix as a form of scaling matrix (simplified version)
    diagonal_matrix = np.diag([scale, scale, 1.0])  # Scale x, y and homogeneous coordinate

    # Combine into a single transformation matrix (matrix multiplication)
    affine_transformation = np.dot(rotation_matrix, diagonal_matrix)
    return affine_transformation[:2] 

def apply_blur_kernel(image, transformation_matrix):
    # Since we're doing a convolution, we need a proper kernel. We'll simulate this by blurring with a basic low-pass filter.
    # The transformation matrix is not directly used in filter2D; instead, we use it to simulate the movement blur effect.
    kernel_size = int(abs(transformation_matrix[0, 0]) + abs(transformation_matrix[1, 1]))  # Kernel size from transformation scale
    kernel_size = max(1, kernel_size)  # Ensure kernel size is at least 1
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
    blurred_image = cv2.filter2D(image, -1, kernel)
    return blurred_image


def add_gaussian_noise(image, mean=0, var=0.01):
    """
    Adds Gaussian noise to an image.
    :param image: Input image
    :param mean: Mean of the Gaussian noise
    :param var: Variance of the Gaussian noise
    :return: Noisy image
    """
    row, col, ch = image.shape
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy_image = image + gauss
    noisy_image = np.clip(noisy_image, 0, 255)  # Clip to ensure valid pixel values
    return noisy_image.astype(np.uint8)

def blur_simulation_and_save(file_path, output_path, dmin, dmax, amin, amax, add_noise=False):
    # Load image
    image = load_image(file_path)
    
    # Generate random degree and angle for blur
    degree, angle = generate_random_degree_and_angle(dmin, dmax, amin, amax)
    
    # Create blur kernel
    kernel = create_blur_kernel(degree, angle)
    
    # Apply blur kernel to image
    blurred_image = apply_blur_kernel(image, kernel)

    # Optionally add Gaussian noise
    if add_noise:
        blurred_image = add_gaussian_noise(blurred_image)

    # Save the blurred image to the specified output path
    cv2.imwrite(output_path, blurred_image)

    return output_path



output_file_path = blur_simulation_and_save('/home/thomas/Sourav_seminar/MedDeblur-main/brain_mri_dataset/ID_0000_AGE_0060_CONTRAST_1_CT.tif', '/home/thomas/Sourav_seminar/MedDeblur-main/MID-DRAN/testingImages/sampleImages/blurred_CT_scan.png', -45, 45, -45, 45,add_noise=True)
