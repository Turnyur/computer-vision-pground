# Computer Vision

This repository serves as a hands-on playground, where I implemented various computer vision techniques covered in my CV course at FAU, Germany. The lecture is based on Richard Szeliski's book: **"Computer Vision: Algorithms and Applications"** and **"Multiple View Geometry in Computer Vision"** by Richard Hartley and Andrew Zisserman. Here I look into fundamental and advanced topics of computer vision.

## Overview

The exercises cuts across wide range of computer vision tasks, where I focus on both theory and practice. The goal of this repository is to:

- **Solidify theoretical concepts** by coding them from scratch.
- **Experiment with my personal images**.
- **Explore advanced algorithms and techniques** used in modern computer vision pipelines.

## Topics Covered

### Image Processing and Feature Detection

- Use of **OpenCV** for image manipulation and processing.
- **Gaussian Filtering**, **Edge Detection** using Canny, and **Image Thresholding**.
- **Feature Detection** using techniques such as **Harris Corners**, and **Oriented FAST and Rotated BRIEF (ORB)**.
- **Feature Extraction** for identifying key points in images.

### Panorama Stitching

- Techniques for combining multiple overlapping images into a seamless panorama.
- **Feature Matching** and **Descriptor-Outlier Removal** using algorithms like SIFT and ORB.
- **Homography estimation** with **RANSAC** for aligning images.

### Structure from Motion (SfM)

- Reconstruction of 3D structures from 2D image sequences.
- **Identification of Inliers**: Leveraging the **epipolar geometry** and **distance from the epipolar line**.
- Application of the **8-Point Algorithm** for computing the **Fundamental Matrix**.
- **Triangulation** to recover the 3D position of matched key points from two or more views.

### Camera Models and Essential Matrix

- Decomposing the **Essential Matrix** to retrieve the camera's rotation and translation.
- Working with camera calibration techniques and **Projection Matrices**.

### Optical Flow and Image Alignment

- Implementation of **Lucas-Kanade Optical Flow** for tracking motion in video sequences.
- **Iterative solvers** to improve motion estimation between frames.
- Use of **Bilinear Interpolation** for sub-pixel accuracy in optical flow calculations.

### Stereo Vision and Disparity Maps

- Calculation of **Disparity Maps** for depth estimation from stereo images.
- Techniques for estimating the **maximum translation** between stereo pairs.
- **Rendering disparity hypotheses** by comparing pixel shifts between left and right stereo images.
- Application of **3D filtering** and post-processing techniques to enhance disparity maps and ensure local coherence.

## Jupyter Notebooks

In addition to the main implementation files in `src`, `utils`, and `resource`, I have also created **Jupyter notebooks** as a scratchpad for experimenting with various algorithms and concepts from the lectures. These notebooks serve as an interactive way to:

- Test out different methods.
- Visualize outputs and intermediary results.
- Debug and refine implementations.

While the notebooks are primarily for experimenting and trying out ideas with my own custom images, the main exercise implementations are structured in the organized directories:

- **src**: Core implementations of the algorithms.
- **utils**: Helper functions for processing images and computations.
- **resource**: Contains datasets and images used in the exercises.

By visualizing and comparing results in the notebooks, I’ve been able to gain valuable insights into the effectiveness of these techniques, which has been crucial for understanding lots of computer vision algorithms.

### Notable Implementations:

- **Feature detection** and matching to align and stitch images into panoramas.
- **Homography estimation** with outlier rejection using RANSAC for robust alignment.
- **Structure from Motion** with triangulation and camera pose recovery to reconstruct 3D scenes.
- **Optical Flow** for tracking motion in sequences of images.
- **Disparity Map Computation** for extracting depth information from stereo image pairs.
