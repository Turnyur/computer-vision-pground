import cv2
import sys
import numpy as np
import scipy.ndimage
import scipy.signal
import scipy.spatial as spatial
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt

# These are type hints, they mostly make the code readable and testable
t_img = np.array
t_disparity = np.array
t_points = np.array
t_descriptors = np.array


def extract_features(img: t_img, num_features: int = 500) -> Tuple[t_points, t_descriptors]:
    """Extract keypoints and their descriptors.
    The OpenCV implementation of ORB is used as a backend.
    https://en.wikipedia.org/wiki/Oriented_FAST_and_rotated_BRIEF
    fast robust local feature detector, It is based on the FAST keypoint detector and a modified version of the visual descriptor BRIEF (Binary Robust Independent Elementary Features).
    Its aim is to provide a fast and efficient alternative to SIFT.

    Args:
        img: a numpy array of [H x Wx 3] size with byte values.
        num_features: an integer signifying how many points we desire.

    Returns:
        A tuple containing a numpy array of [N x 2] and numpy array of [N x 32]
    """
    
    #TODO : Hint - you will need cv2.ORB_create. 
    #TODO You have already implmented this function in Exercise 2: Panorama Stitching.
    ... 
    if len(img.shape)==3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
            gray_img = img


    orb = cv2.ORB_create(nfeatures=num_features)
    keypoints, descriptors = orb.detectAndCompute(gray_img, None)
    keypoints_array = np.array([kp.pt for kp in keypoints])


    return keypoints_array, descriptors


def filter_and_align_descriptors(f1: Tuple[t_points, t_descriptors], f2: Tuple[t_points, t_descriptors],
                                 similarity_threshold=.7, similarity_metric='hamming') -> Tuple[t_points, t_points]:
    """Aligns pairs of keypoints from two images.
    Aligns keypoints from two images based on descriptor similarity.
    If K points have been detected in image1 and J points have been detected in image2, the result will be to sets of N
    points representing points with similar descriptors; where N <= J and K <=points.

    Args:
        f1: A tuple of two numpy arrays with the first array having dimensions [N x 2] and the second one [N x M]. M
            representing the dimensionality of the point features. In the case of ORB features, M is 32.
        f2: A tuple of two numpy arrays with the first array having dimensions [J x 2] and the second one [J x M]. M
            representing the dimensionality of the point features. In the case of ORB features, M is 32.
        similarity_threshold: The ratio the distance of most similar descriptor in image2 to the distance of the second
            most similar ratio.
        similarity_metric: A string with the name of the metric by which distances are calculated. It must be compatible
            with the ones that are defined for scipy.spatial.distance.cdist.

    Returns:
        A tuple of numpy arrays both sized [ N x 2]representing the similar point locations.

    """
    #TODO You have already implmented this function in Exercise 2: Panorama Stitching.

    keypoint1, description1 = f1
    keypoint2, description2 = f2

    if similarity_metric == 'hamming':
        dist_matrix = np.zeros((description1.shape[0], description2.shape[0]), dtype=np.float32) # NXN matrix
        
        for i in range(description1.shape[0]):
            dist_matrix[i, :] = np.sum(np.bitwise_xor(description1[i], description2) != 0, axis=1) / description1.shape[1]
        
    else:
        raise ValueError("Invalid similarity metric!")

    # step 2: computing the indexes of src dst so that src[src_idx,:] and dst[dst,:] refer to matching points.
    sorted_indices = np.argsort(dist_matrix, axis=1)
    best_matches = sorted_indices[:, 0]
    second_best_matches = sorted_indices[:, 1]

    # step 3: find a boolean index of the matched pairs that is true only if a match was significant.
    # A match is considered significant if the ratio of its distance to the second best is lower than a given
    # threshold.
    # Hint: use the previously computed distance matrix to find the second best match.
    distances = dist_matrix[np.arange(description1.shape[0]), best_matches]
    second_best_distances = dist_matrix[np.arange(description1.shape[0]), second_best_matches]
    ratio = distances / second_best_distances # ratio of significance
    significant_matches = ratio < similarity_threshold # Boolean of matched

    # step 4: removing non-significant matches and return the aligned points (their location only!)
    src_indices = np.arange(description1.shape[0])[significant_matches]
    dst_indices = best_matches[significant_matches]

    aligned_kp1 = keypoint1[src_indices]
    aligned_kp2 = keypoint2[dst_indices]


    return aligned_kp1, aligned_kp2

def get_max_translation(src: t_img, dst: t_img, well_aligned_thr=.1) -> int:
    """finds the maximum translation/shift between two images

    Args:
        src: one image taken from a camera, numpy array of shape [H x W x 3]
        dst: another image with camera only translate, numpy array of shape [H x W x 3]
        well_aligned_thr: a float representing the maximum y wise distance between valid matching points.

    Returns:
        an integer value representing the maximum translation of the camera from src to dst image
    """

    # Step 1: Generate features/descriptors, filter and align them (courtesy of exercise 2)
    keypoint_and_desc1 = extract_features(src)
    keypoint_and_desc2 = extract_features(dst)

    # Step 2: filter out correspondences that are not horizontally aligned using well aligned threshold
    f1, f2 = filter_and_align_descriptors(keypoint_and_desc1, keypoint_and_desc2)

    # Step 3: Find the translation across the image using the descriptors and return the maximum value
    max_translation = 0
    N = f1.shape[0]
    for i in range(N):
        x1, y1 = f1[i]
        x2, y2 = f2[i]
        if np.abs(y2 -y1)<well_aligned_thr:
            dist = np.abs(x2 - x1)
            max_translation = np.maximum(dist, max_translation)
        

    return np.round(max_translation)


def render_disparity_hypothesis(src: t_img, dst: t_img, offset: int, pad_size: int) -> t_disparity:
    """Calculates the agreement between the shifted src image and the dst image.

    Args:
        src: one image taken from a camera, numpy array of shape [H x W x 3]
        dst: another image with camera only translate, numpy array of shape [H x W x 3]
        offset: an integer value by which the image is shifted
        pad_size: an integer value to pad the images for computation

    Returns:
        a numpy array of shape [H x W] containing the euclidean distance between RGB values of the shifted src and dst
        images.
    """

    # Step 1: Pad necessary values to src and dst
    src_padded = np.pad(src, ((0, 0), (pad_size, pad_size), (0, 0)), mode='constant', constant_values=0)
    
    
    # Step 2: Shift src by the offset (to the right) to create a disparity hypothesis
    src_shifted = src_padded[:, pad_size + offset: pad_size + offset + dst.shape[1], :]
    
    
    # Step 3: Compute the Euclidean distance between the shifted src and the dst image
    # print("SHAPE: SRC_SHIFTED: ", src_shifted.shape)
    # print("SHAPE: DST: ", dst.shape)
    # print("\n\n")

   # Step 3: Plot and save the shifted src image with axis on and white background
    # plt.imshow(src_shifted)
    # plt.axis('on')  # Ensure axis is on
    # plt.savefig(f"./output/src_shifted_{offset}.png", facecolor='white')  # Save with white background
    # plt.close()  # Close the figure to avoid overlap
    
    # # Step 4: Plot and save the dst image with axis on and white background
    # plt.imshow(dst)
    # plt.axis('on')  # Ensure axis is on
    # plt.savefig(f"./output/dst_{offset}.png", facecolor='white')  # Save with white background
    # plt.close()  # Close the figure to avoid overlap 
    

    disparity = np.linalg.norm(src_shifted - dst, axis=2)
    #print("SHAPE: DISPARITY: ", disparity.shape)


    return disparity

def disparity_map(src: t_img, dst: t_img, offset: int, pad_size: int, sigma_x: int, sigma_z: int,
                  median_filter_size: int) -> t_disparity:
    """calculates the best/minimum disparity map for a given pair of images

    Args:
        src: one image taken from a camera, numpy array of shape [H x W x 3]
        dst: another image with camera only translate, numpy array of shape [H x W x 3]
        offset: an integer value by which the image is shifted
        pad_size: an integer value to pad the images for computation
        sigma_x: an integer value for standard deviation in x-direction for gaussian filter
        sigma_z: an integer value for standard deviation in z-direction for gaussian filter
        median_filter_size: an integer value representing the window size for applying median filter

    Returns:
        a numpy array of shape [H x W] containing the minimum/best disparity values for a pair of images
    """

    # Step 1: Construct a stack of all reasonable disparity hypotheses.

    # Step 2: Enforce the coherence between x-axis and disparity-axis using a 3D gaussian filter onto
    # the stack of disparity hypotheses

    # Step 3: Choose the best disparity hypothesis for every pixel

    # Step 4: Apply the median filter to enhance local consensus

    disparity_hypotheses = np.zeros((src.shape[0], src.shape[1], offset + 1))
    
    for d in range(offset + 1):
        disparity_hypotheses[:, :, d] = render_disparity_hypothesis(src, dst, d, pad_size)

    # Step 2: Enforce the coherence between x-axis and disparity-axis using a 3D Gaussian filter.
    # The Gaussian filter smooths the disparity hypotheses across both the pixel positions and disparity values.
    #smoothed_hypotheses = scipy.ndimage.gaussian_filter(disparity_hypotheses, sigma=(sigma_x, sigma_x, sigma_z))
    # 0 for y-direction
    
    # smoothed_hypotheses = scipy.ndimage.gaussian_filter(disparity_hypotheses, sigma=(0, sigma_x, sigma_z), mode='nearest')

    

    # # Step 3: Choose the best disparity hypothesis for every pixel.
    # best_disparity_indices = np.argmin(smoothed_hypotheses, axis=2)
    # best_disparity_map = best_disparity_indices.astype(np.float32)

    # # Step 4: Apply the median filter to enhance local consensus.
    # best_disparity_map = scipy.ndimage.median_filter(best_disparity_map, size=median_filter_size)

    # #print("SHAPE: BEST_DISPARITY: ", np.max(best_disparity_map))
    # print("SHAPE: BEST_DISPARITY: ", np.max(best_disparity_map))
    # best = 

    # return best_disparity_map
    smoothed_hypotheses = scipy.ndimage.gaussian_filter(disparity_hypotheses, sigma=(0, sigma_x, sigma_z), mode='nearest')

    # Step 3: Choose the best disparity hypothesis for every pixel (minimum cost).
    best_disparity_indices = np.argmin(smoothed_hypotheses, axis=2)

    # Create an array of actual disparity values
    disparity_values = np.arange(offset + 1)

    # Map indices to actual disparity values
    actual_disparity_map = disparity_values[best_disparity_indices]

    # Step 4: Apply the median filter to enhance local consensus.
    actual_disparity_map = scipy.ndimage.median_filter(actual_disparity_map, size=median_filter_size)

    return actual_disparity_map


def bilinear_grid_sample(img: t_img, x_array: t_img, y_array: t_img) -> t_img:
    """Sample an image according to a sampling vector field.

    Args:
        img: one image, numpy array of shape [H x W x 3]
        x_array: a numpy array of [H' x W'] representing the x coordinates src x-direction
        y_array: a numpy array of [H' x W'] representing interpolation in y-direction

    Returns:
        An image of size [H' x W'] containing the sampled points in
    """

    # Step 1: Estimate the left, top, right, bottom integer parts (l, r, t, b)
    # and the corresponding coefficients (a, b, 1-a, 1-b) of each pixel

    # Step 2: Take care of out of image coordinates

    # Step 3: Produce a weighted sum of each rounded corner of the pixel

    # Step 4: Accumulate and return all the weighted four corners

    H, W, C = img.shape
    H_prime, W_prime = x_array.shape
    
    # Step 1: Estimate the left, top, right, bottom integer parts and coefficients
    l = np.floor(x_array).astype(int)  # left
    r = l + 1                           # right
    t = np.floor(y_array).astype(int)  # top
    b = t + 1                           # bottom
    
    a = x_array - l                     # horizontal coefficient
    b = y_array - t                     # vertical coefficient
    
    # Step 2: Take care of out-of-image coordinates
    l = np.clip(l, 0, W - 1)
    r = np.clip(r, 0, W - 1)
    t = np.clip(t, 0, H - 1)
    b = np.clip(b, 0, H - 1)

    # Step 3: Produce a weighted sum of each rounded corner of the pixel
    # Access the pixel values
    Ia = img[t, l] #if t < H and l < W else np.zeros((C,))
    Ib = img[t, r] #if b < H and l < W else np.zeros((C,))
    Ic = img[b, l] #if t < H and r < W else np.zeros((C,))
    Id = img[b, r] #if b < H and r < W else np.zeros((C,))
    
    # Weighted sum for each channel
    sampled_img = (1 - a) * (1 - b) * Ia + \
                  a *(1 - a) * Ib + \
                  (1 - b) * b * Ic + \
                  a * b * Id

    plt.imshow(sampled_img)
    plt.axis('on')  # Ensure axis is on
    plt.savefig(f"./output/sampled_image.png", facecolor='white')  # Save with white background
    plt.close()  # Close the figure to avoid overlap 

    plt.imshow(img)
    plt.axis('on')  # Ensure axis is on
    plt.savefig(f"./output/orig_image.png", facecolor='white')  # Save with white background
    plt.close()  # Close the figure to avoid overlap 
    
    print("SAMPLED_IMAGE: ", img)
    
    return sampled_img

#
