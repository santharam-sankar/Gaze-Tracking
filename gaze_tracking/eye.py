"""
Eye Module

This module provides the Eye class that isolates eye regions from
facial landmarks and performs eye-specific analysis including
blink detection and pupil isolation.
"""

import math
import numpy as np
import cv2
from .pupil import Pupil


class Eye(object):
    """
    Represents a single eye and handles eye region isolation and analysis.
    
    This class:
    - Isolates the eye region from the full face frame
    - Calculates blinking ratio (eye aspect ratio)
    - Coordinates pupil detection within the isolated eye region
    
    The 68-point facial landmark model defines specific points for each eye:
    - Left eye: points 36-41
    - Right eye: points 42-47
    """

    # Facial landmark indices for left eye (from 68-point model)
    LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]
    
    # Facial landmark indices for right eye (from 68-point model)
    RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]

    def __init__(self, original_frame, landmarks, side, calibration):
        """Initialize an Eye object and perform analysis.
        
        Arguments:
            original_frame (numpy.ndarray): Grayscale frame containing the face
            landmarks (dlib.full_object_detection): 68-point facial landmarks
            side (int): 0 for left eye, 1 for right eye
            calibration (Calibration): Calibration object for threshold optimization
        """
        self.frame = None  # Isolated eye region frame
        self.origin = None  # (x, y) coordinates of top-left corner of eye region in original frame
        self.center = None  # (x, y) center point of the eye region
        self.pupil = None  # Pupil object containing detected pupil coordinates
        self.landmark_points = None  # Array of landmark points for this eye

        # Perform eye analysis: isolate region, detect pupil, calculate blink ratio
        self._analyze(original_frame, landmarks, side, calibration)

    @staticmethod
    def _middle_point(p1, p2):
        """Returns the middle point (x,y) between two points

        Arguments:
            p1 (dlib.point): First point
            p2 (dlib.point): Second point
        """
        x = int((p1.x + p2.x) / 2)
        y = int((p1.y + p2.y) / 2)
        return (x, y)

    def _isolate(self, frame, landmarks, points):
        """Isolate the eye region from the full face frame.

        This method:
        1. Extracts landmark points for the eye
        2. Creates a mask to isolate only the eye region
        3. Crops the frame to a bounding box around the eye
        4. Calculates the center and origin coordinates

        Arguments:
            frame (numpy.ndarray): Grayscale frame containing the face
            landmarks (dlib.full_object_detection): Facial landmarks for the face region
            points (list): Landmark point indices for this eye (from 68-point model)
        """
        # Extract (x, y) coordinates for all eye landmark points
        region = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in points])
        region = region.astype(np.int32)
        self.landmark_points = region

        # Create a mask to isolate only the eye region
        # This removes other facial features from the eye frame
        height, width = frame.shape[:2]
        black_frame = np.zeros((height, width), np.uint8)
        mask = np.full((height, width), 255, np.uint8)
        # Fill the eye region with black in the mask
        cv2.fillPoly(mask, [region], (0, 0, 0))
        # Apply mask to extract only the eye region
        eye = cv2.bitwise_not(black_frame, frame.copy(), mask=mask)

        # Crop to a bounding box around the eye with a small margin
        margin = 5  # Pixels of padding around the eye
        min_x = np.min(region[:, 0]) - margin
        max_x = np.max(region[:, 0]) + margin
        min_y = np.min(region[:, 1]) - margin
        max_y = np.max(region[:, 1]) + margin

        # Extract the eye region as a separate frame
        self.frame = eye[min_y:max_y, min_x:max_x]
        # Store origin for converting relative coordinates back to absolute
        self.origin = (min_x, min_y)

        # Calculate center point of the isolated eye region
        height, width = self.frame.shape[:2]
        self.center = (width / 2, height / 2)

    def _blinking_ratio(self, landmarks, points):
        """Calculate the eye aspect ratio (EAR) to detect blinking.

        The Eye Aspect Ratio (EAR) is the ratio of eye width to eye height.
        When eyes are open, the ratio is relatively low. When eyes are closed,
        the height decreases significantly, causing the ratio to increase.

        Formula: EAR = eye_width / eye_height

        Arguments:
            landmarks (dlib.full_object_detection): Facial landmarks for the face region
            points (list): Landmark point indices for this eye

        Returns:
            float: The computed EAR ratio, or None if calculation fails
        """
        # Get horizontal eye corners (leftmost and rightmost points)
        left = (landmarks.part(points[0]).x, landmarks.part(points[0]).y)
        right = (landmarks.part(points[3]).x, landmarks.part(points[3]).y)
        
        # Get vertical eye boundaries (top and bottom midpoints)
        top = self._middle_point(landmarks.part(points[1]), landmarks.part(points[2]))
        bottom = self._middle_point(landmarks.part(points[5]), landmarks.part(points[4]))

        # Calculate Euclidean distances
        eye_width = math.hypot((left[0] - right[0]), (left[1] - right[1]))
        eye_height = math.hypot((top[0] - bottom[0]), (top[1] - bottom[1]))

        try:
            # Higher ratio = more closed eye
            ratio = eye_width / eye_height
        except ZeroDivisionError:
            # Avoid division by zero if eye height is 0
            ratio = None

        return ratio

    def _analyze(self, original_frame, landmarks, side, calibration):
        """Perform complete eye analysis: isolation, calibration, and pupil detection.

        This method:
        1. Selects the appropriate landmark points based on eye side
        2. Calculates blinking ratio
        3. Isolates the eye region
        4. Updates calibration if needed
        5. Detects the pupil using the calibrated threshold

        Arguments:
            original_frame (numpy.ndarray): Grayscale frame containing the face
            landmarks (dlib.full_object_detection): Facial landmarks for the face region
            side (int): 0 for left eye, 1 for right eye
            calibration (calibration.Calibration): Calibration manager for threshold optimization
        """
        # Select landmark points based on which eye we're processing
        if side == 0:
            points = self.LEFT_EYE_POINTS
        elif side == 1:
            points = self.RIGHT_EYE_POINTS
        else:
            return  # Invalid side value

        # Calculate blinking ratio (eye aspect ratio)
        self.blinking = self._blinking_ratio(landmarks, points)
        
        # Isolate the eye region from the full face frame
        self._isolate(original_frame, landmarks, points)

        # Update calibration if not yet complete (needs 20 frames per eye)
        # Calibration finds the optimal threshold for binarization
        if not calibration.is_complete():
            calibration.evaluate(self.frame, side)

        # Get the calibrated threshold for this eye
        threshold = calibration.threshold(side)
        
        # Detect the pupil using the isolated eye frame and threshold
        self.pupil = Pupil(self.frame, threshold)
