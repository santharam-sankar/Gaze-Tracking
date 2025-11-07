"""
Pupil Module

This module provides the Pupil class that detects the iris/pupil
within an isolated eye region using image processing techniques.
"""

import numpy as np
import cv2


class Pupil(object):
    """
    Detects the pupil/iris position within an isolated eye frame.
    
    Uses image processing techniques including:
    - Bilateral filtering for noise reduction
    - Erosion for morphological operations
    - Thresholding for binarization
    - Contour detection to find the iris
    - Centroid calculation to locate the pupil center
    """

    def __init__(self, eye_frame, threshold):
        """Initialize pupil detection for an eye frame.
        
        Arguments:
            eye_frame (numpy.ndarray): Isolated grayscale eye region
            threshold (int): Binarization threshold value (0-255)
        """
        self.iris_frame = None  # Processed binary frame showing iris
        self.threshold = threshold  # Threshold value for binarization
        self.x = None  # X coordinate of pupil center (relative to eye frame)
        self.y = None  # Y coordinate of pupil center (relative to eye frame)

        # Perform iris detection immediately
        self.detect_iris(eye_frame)

    @staticmethod
    def image_processing(eye_frame, threshold):
        """Process the eye frame to isolate the iris region.

        Processing pipeline:
        1. Bilateral filtering: Reduces noise while preserving edges
        2. Erosion: Removes small artifacts and smooths the iris boundary
        3. Thresholding: Creates binary image (iris = black, rest = white)

        Arguments:
            eye_frame (numpy.ndarray): Grayscale frame containing isolated eye region
            threshold (int): Threshold value (0-255) for binarization

        Returns:
            numpy.ndarray: Binary frame where iris appears as black region
        """
        # Small kernel for morphological operations
        kernel = np.ones((3, 3), np.uint8)
        
        # Bilateral filter: preserves edges while reducing noise
        # Parameters: (image, diameter, sigmaColor, sigmaSpace)
        new_frame = cv2.bilateralFilter(eye_frame, 10, 15, 15)
        
        # Erosion: removes small artifacts and smooths boundaries
        new_frame = cv2.erode(new_frame, kernel, iterations=3)
        
        # Binary threshold: pixels below threshold become 0 (black), above become 255 (white)
        # The iris (darker) should become black, rest becomes white
        new_frame = cv2.threshold(new_frame, threshold, 255, cv2.THRESH_BINARY)[1]

        return new_frame

    def detect_iris(self, eye_frame):
        """Detect the iris and calculate its center position (pupil center).

        The method:
        1. Processes the eye frame to isolate the iris
        2. Finds contours in the binary image
        3. Selects the second-largest contour (iris, excluding frame edges)
        4. Calculates the centroid (center of mass) as the pupil position

        Arguments:
            eye_frame (numpy.ndarray): Grayscale frame containing isolated eye region
        """
        # Process the frame to get binary iris image
        self.iris_frame = self.image_processing(eye_frame, self.threshold)

        # Find all contours in the binary image
        # RETR_TREE: retrieves all contours and reconstructs hierarchy
        # CHAIN_APPROX_NONE: stores all contour points
        contours, _ = cv2.findContours(self.iris_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
        
        # Sort contours by area (largest first)
        # The largest is usually the frame boundary, second-largest is the iris
        contours = sorted(contours, key=cv2.contourArea)

        try:
            # Use the second-largest contour (index -2) as the iris
            # Calculate image moments to find the centroid
            moments = cv2.moments(contours[-2])
            
            # Centroid formula: (m10/m00, m01/m00)
            # m00 = total area, m10/m01 = weighted position
            self.x = int(moments['m10'] / moments['m00'])
            self.y = int(moments['m01'] / moments['m00'])
        except (IndexError, ZeroDivisionError):
            # If no suitable contour found or division by zero, leave coordinates as None
            pass
