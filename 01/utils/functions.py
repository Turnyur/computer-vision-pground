import cv2
import numpy as np
from typing import Tuple


def compute_harris_response(I: np.array, k: float = 0.06) -> Tuple[np.array]:
    """Determines the Harris Response of an Image.

    Args:
        I: A Gray-level image in float32 format.
        k: A constant changing the trace to determinant ratio.

    Returns:
        A tuple with float images containing the Harris response (R) and other intermediary images. Specifically
        (R, A, B, C, Idx, Idy).
    """
    assert I.dtype == np.float32

    # Step 1: Compute Idx and Idy with cv2.Sobel
    Idx = cv2.Sobel(I, cv2.CV_32F, 1, 0, ksize=3)
    Idy = cv2.Sobel(I, cv2.CV_32F, 0, 1, ksize=3)

    # Step 2: Ixx Iyy Ixy from Idx and Idy
    Ixx = pow(Idx, 2)
    Iyy = pow(Idy, 2)
    Ixy = Idx * Idy

    # Step 3: compute A, B, C from Ixx, Iyy, Ixy with cv2.GaussianBlur
    # Use sdev = 1 and kernelSize = (3, 3) in cv2.GaussianBlur
    sdev = 1
    kernelSize = (3, 3)
    A = cv2.GaussianBlur(Ixx, kernelSize, sdev)
    B = cv2.GaussianBlur(Iyy, kernelSize, sdev)
    C = cv2.GaussianBlur(Ixy, kernelSize, sdev)

    # Step 4: Compute the harris response with the determinant and the trace of T
    det_T = A * B - C * C
    trace_T = A + B
    R = det_T - k * (trace_T**2)

    return (R, A, B, C, Idx, Idy)


def detect_corners_old(R: np.array, threshold: float = 0.1) -> Tuple[np.array, np.array]:
    """Computes key-points from a Harris response image.

    Key points are all points where the harris response is significant and greater than its neighbors.

    Args:
        R: A float image with the harris response
        threshold: A float determining which Harris response values are significant.

    Returns:
        A tuple of two 1D integer arrays containing the x and y coordinates of key-points in the image.
    """
    # Step 1 (recommended): Pad the response image to facilitate vectorization
    padded_R = np.pad(R, pad_width=1, mode='constant', constant_values=0)
    corners = np.zeros(R.shape[:2], dtype=bool)  # Only 2D array needed for corner detection
    
    # Get the shape of the Harris response function
    rows, cols = R.shape

    # Iterate over each pixel in R
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            # Check the threshold for all channels
            pixel_values = padded_R[i, j]
            if np.all(pixel_values > threshold):
                # Extract the 3x3 neighborhood for each channel
                neighborhood = padded_R[i-1:i+2, j-1:j+2]
                
                # Check if the current pixel is a local maximum across all channels
                if np.all(pixel_values == np.max(neighborhood, axis=(0, 1))):
                    corners[i, j] = True

    # Get the coordinates of the detected corners
    



    # Step 2 (recommended): Create one image for every offset in the 3x3 neighborhood


    # Step 3 (recommended): Compute the greatest neighbor of every pixel


    # Step 4 (recommended): Compute a boolean image with only all key-points set to True


    # Step 5 (recommended): Use np.nonzero to compute the locations of the key-points from the boolean image
    y_coords, x_coords = np.nonzero(corners)

    return (y_coords, x_coords)



def detect_corners(R: np.array, threshold: float = 0.1) -> Tuple[np.array, np.array]:
    """Computes key-points from a Harris response image.

    Key points are all points where the Harris response is significant and greater than its neighbors.

    Args:
        R: A float image with the Harris response.
        threshold: A float determining which Harris response values are significant.

    Returns:
        A tuple of two 1D integer arrays containing the x and y coordinates of key-points in the image.
    """
    # Step 1 (recommended): Pad the response image to facilitate vectorization
    padded_R = np.pad(R, pad_width=1, mode='constant', constant_values=0)
    corners = np.zeros(R.shape[:2], dtype=bool)  # Only 2D array needed for corner detection
    
    # Step 2 (recommended): Create one image for every offset in the 3x3 neighborhood
    neighborhoods = np.stack([padded_R[1:-1, 1:-1],       # center
                              padded_R[:-2, :-2],         # top-left
                              padded_R[:-2, 1:-1],        # top-center
                              padded_R[:-2, 2:],          # top-right
                              padded_R[1:-1, :-2],        # center-left
                              padded_R[1:-1, 2:],         # center-right
                              padded_R[2:, :-2],          # bottom-left
                              padded_R[2:, 1:-1],         # bottom-center
                              padded_R[2:, 2:]], axis=0)  # bottom-right
    
    # Step 3 (recommended): Compute the greatest neighbor of every pixel
    neighbors_max = np.max(neighborhoods[1:], axis=0)  # Skip the center image for comparison
    
    # Step 4 (recommended): Compute a boolean image with only key-points set to True
    # A pixel is a corner if it is greater than the threshold and greater than its neighbors
    corners = (R > neighbors_max) & (R > threshold)
    
    # Step 5 (recommended): Use np.nonzero to compute the locations of the key-points from the boolean image
    y_coords, x_coords = np.nonzero(corners)

    return (y_coords, x_coords)


def detect_edges_old(R: np.array, edge_threshold: float = -0.01) -> np.array:
    """Computes a boolean image where edge pixels are set to True.

    Edges are significant pixels of the harris response that are a local minimum along the x or y axis.

    Args:
        R: a float image with the harris response.
        edge_threshold: A constant determining which response pixels are significant

    Returns:
        A boolean image with edge pixels set to True.
    """
    # Step 1 (recommended): Pad the response image to facilitate vectorization
    padded_R = np.pad(R, pad_width=1, mode='constant', constant_values=0)
    edges = np.zeros(R.shape[:2], dtype=bool)  # Only 2D array needed for corner detection

    # Get the shape of the Harris response function
    rows, cols = R.shape

    # Iterate over each pixel in R
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            # Check the threshold for all channels
            pixel_values = padded_R[i, j]
            if np.all(pixel_values <= edge_threshold):
                # Extract the 3x3 neighborhood for each channel
                neighborhood = padded_R[i-1:i+2, j-1:j+2]
                
                # Check if the current pixel is a local maximum across all channels
                if np.all(pixel_values == np.min(neighborhood, axis=(0, 1))):
                    edges[i, j] = True


    # Step 2 (recommended): Calculate significant response pixels


    # Step 3 (recommended): Create two images with the smaller x-axis and y-axis neighbors respectively


    # Step 4 (recommended): Calculate pixels that are lower than either their x-axis or y-axis neighbors


    # Step 5 (recommended): Calculate valid edge pixels by combining significant and axis_minimal pixels


    return edges

def detect_edges(R: np.array, edge_threshold: float = -0.01) -> np.array:
    """Computes a boolean image where edge pixels are set to True.

    Edges are significant pixels of the harris response that are a local minimum along the x or y axis.

    Args:
        R: A float image with the Harris response.
        edge_threshold: A constant determining which response pixels are significant.

    Returns:
        A boolean image with edge pixels set to True.
    """
    # Step 1 (recommended): Pad the response image to facilitate vectorization
    padded_R = np.pad(R, pad_width=1, mode='constant', constant_values=0)

    # Step 2: Calculate significant response pixels
    # A pixel is significant if it is less than or equal to the edge_threshold
    significant_pixels = R <= edge_threshold

    # Step 3: Create two images with the smaller x-axis and y-axis neighbors respectively
    # Get shifted versions of the padded image to compare x and y neighbors
    R_left = padded_R[1:-1, :-2]  # Left neighbor
    R_right = padded_R[1:-1, 2:]  # Right neighbor
    R_up = padded_R[:-2, 1:-1]    # Top neighbor
    R_down = padded_R[2:, 1:-1]   # Bottom neighbor

    # Step 4: Calculate pixels that are lower than either their x-axis or y-axis neighbors
    x_axis_minimal = (R <= R_left) & (R <= R_right)
    y_axis_minimal = (R <= R_up) & (R <= R_down)

    # Step 5: Calculate valid edge pixels by combining significant and axis_minimal pixels
    edges = significant_pixels & (x_axis_minimal | y_axis_minimal)

    return edges
    
