"""
Calibration Module

This module provides the Calibration class that optimizes the
binarization threshold for pupil detection. The threshold varies
based on lighting conditions, skin tone, and camera settings.
"""

from __future__ import division
import cv2
from .pupil import Pupil


class Calibration(object):
    """
    Calibrates the pupil detection algorithm by finding optimal threshold values.
    
    The calibration process:
    1. Collects threshold values from multiple frames (20 per eye)
    2. Tests different threshold values to find the one that produces
       an iris size closest to the expected average (48% of eye area)
    3. Averages the collected thresholds for stable detection
    
    This ensures the binarization threshold works well for the specific
    user's eye color, lighting conditions, and camera settings.
    """

    def __init__(self):
        """Initialize the calibration system."""
        self.nb_frames = 20  # Number of frames to collect for calibration
        self.thresholds_left = []  # Collected thresholds for left eye
        self.thresholds_right = []  # Collected thresholds for right eye

    def is_complete(self):
        """Check if calibration has collected enough samples.
        
        Returns:
            bool: True if both eyes have collected nb_frames samples
        """
        return len(self.thresholds_left) >= self.nb_frames and len(self.thresholds_right) >= self.nb_frames

    def threshold(self, side):
        """Get the calibrated threshold value for an eye.
        
        Returns the average of collected thresholds. If calibration is
        incomplete, returns the average of available samples.

        Argument:
            side (int): 0 for left eye, 1 for right eye

        Returns:
            int: Average threshold value (0-255)
        """
        if side == 0:
            # Return average of collected left eye thresholds
            return int(sum(self.thresholds_left) / len(self.thresholds_left))
        elif side == 1:
            # Return average of collected right eye thresholds
            return int(sum(self.thresholds_right) / len(self.thresholds_right))

    @staticmethod
    def iris_size(frame):
        """Calculate the percentage of the eye area occupied by the iris.

        The iris should typically occupy about 48% of the eye area.
        This metric is used to evaluate threshold quality.

        Argument:
            frame (numpy.ndarray): Binary frame (iris = black/0, rest = white/255)

        Returns:
            float: Percentage (0.0-1.0) of frame occupied by iris
        """
        # Crop edges to avoid frame boundary artifacts
        frame = frame[5:-5, 5:-5]
        height, width = frame.shape[:2]
        nb_pixels = height * width
        
        # Count black pixels (iris) - non-zero pixels are white
        nb_blacks = nb_pixels - cv2.countNonZero(frame)
        
        # Return ratio of iris area to total area
        return nb_blacks / nb_pixels

    @staticmethod
    def find_best_threshold(eye_frame):
        """Find the optimal binarization threshold for an eye frame.

        Tests multiple threshold values and selects the one that produces
        an iris size closest to the expected average (48%).

        Argument:
            eye_frame (numpy.ndarray): Grayscale frame of the eye

        Returns:
            int: Optimal threshold value (5-95, in steps of 5)
        """
        # Expected iris size as percentage of eye area
        average_iris_size = 0.48
        trials = {}

        # Test thresholds from 5 to 95 in steps of 5
        for threshold in range(5, 100, 5):
            # Process frame with this threshold
            iris_frame = Pupil.image_processing(eye_frame, threshold)
            # Calculate resulting iris size
            trials[threshold] = Calibration.iris_size(iris_frame)

        # Find threshold that produces iris size closest to expected
        best_threshold, iris_size = min(trials.items(), key=(lambda p: abs(p[1] - average_iris_size)))
        return best_threshold

    def evaluate(self, eye_frame, side):
        """Add a new threshold sample to the calibration data.

        Finds the best threshold for this frame and adds it to the
        collection. After nb_frames samples, the average is used.

        Arguments:
            eye_frame (numpy.ndarray): Grayscale frame of the eye
            side (int): 0 for left eye, 1 for right eye
        """
        # Find optimal threshold for this frame
        threshold = self.find_best_threshold(eye_frame)

        # Store threshold for the appropriate eye
        if side == 0:
            self.thresholds_left.append(threshold)
        elif side == 1:
            self.thresholds_right.append(threshold)
