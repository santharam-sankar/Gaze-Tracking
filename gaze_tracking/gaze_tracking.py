"""
GazeTracking Module

This module provides the main GazeTracking class that coordinates
face detection, eye tracking, and pupil detection to determine
where a user is looking.

The class uses dlib for facial landmark detection and OpenCV for
image processing operations.
"""

from __future__ import division
import os
import cv2
import dlib
from .eye import Eye
from .calibration import Calibration


class GazeTracking(object):
    """
    Main class for tracking user's gaze direction.
    
    This class coordinates the entire gaze tracking pipeline:
    - Face detection using dlib's frontal face detector
    - Facial landmark detection (68-point model)
    - Eye isolation and analysis
    - Pupil detection and coordinate calculation
    - Gaze direction determination (left, right, center)
    - Blink detection
    
    Attributes:
        frame: Current video frame being processed
        eye_left: Left eye object containing detection results
        eye_right: Right eye object containing detection results
        calibration: Calibration object for threshold optimization
    """

    def __init__(self):
        """Initialize the GazeTracking system.
        
        Loads the facial landmark detection model and initializes
        the face detector. The model file must be present in
        gaze_tracking/trained_models/ directory.
        """
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.calibration = Calibration()

        # Initialize dlib's frontal face detector (HOG-based)
        # This is used to detect faces in the video frame
        self._face_detector = dlib.get_frontal_face_detector()

        # Load the facial landmark predictor model
        # This model can detect 68 facial landmarks including eye corners
        cwd = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.abspath(os.path.join(cwd, "trained_models/shape_predictor_68_face_landmarks.dat"))
        self._predictor = dlib.shape_predictor(model_path)

    @property
    def pupils_located(self):
        """Check if pupils have been successfully located in both eyes.
        
        Returns:
            bool: True if both pupils are detected, False otherwise
        """
        try:
            # Try to access pupil coordinates - if they exist, conversion to int will succeed
            int(self.eye_left.pupil.x)
            int(self.eye_left.pupil.y)
            int(self.eye_right.pupil.x)
            int(self.eye_right.pupil.y)
            return True
        except Exception:
            # If any coordinate is None or eye objects don't exist, return False
            return False

    def _analyze(self):
        """Detect face and initialize Eye objects for gaze tracking.
        
        This method:
        1. Converts frame to grayscale (required for dlib)
        2. Detects faces in the frame
        3. Extracts facial landmarks for the first detected face
        4. Creates Eye objects for left and right eyes
        """
        # Convert to grayscale - dlib requires grayscale images
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame (returns list of rectangles)
        faces = self._face_detector(frame)

        try:
            # Get facial landmarks for the first detected face
            # The 68-point model includes eye corners and other facial features
            landmarks = self._predictor(frame, faces[0])
            
            # Initialize Eye objects for left (0) and right (1) eyes
            # Each Eye object isolates the eye region and detects the pupil
            self.eye_left = Eye(frame, landmarks, 0, self.calibration)
            self.eye_right = Eye(frame, landmarks, 1, self.calibration)

        except IndexError:
            # No face detected - set eyes to None
            self.eye_left = None
            self.eye_right = None

    def refresh(self, frame):
        """Process a new frame and update gaze tracking data.

        This should be called for each frame from the video stream.
        It triggers face detection, eye isolation, and pupil detection.

        Arguments:
            frame (numpy.ndarray): The BGR video frame to analyze
        """
        self.frame = frame
        self._analyze()

    def pupil_left_coords(self):
        """Get the coordinates of the left pupil in the original frame.
        
        Returns:
            tuple: (x, y) coordinates of the left pupil, or None if not detected
        """
        if self.pupils_located:
            # Convert from eye-relative coordinates to frame-absolute coordinates
            # origin is the top-left corner of the eye region in the original frame
            x = self.eye_left.origin[0] + self.eye_left.pupil.x
            y = self.eye_left.origin[1] + self.eye_left.pupil.y
            return (x, y)

    def pupil_right_coords(self):
        """Get the coordinates of the right pupil in the original frame.
        
        Returns:
            tuple: (x, y) coordinates of the right pupil, or None if not detected
        """
        if self.pupils_located:
            # Convert from eye-relative coordinates to frame-absolute coordinates
            x = self.eye_right.origin[0] + self.eye_right.pupil.x
            y = self.eye_right.origin[1] + self.eye_right.pupil.y
            return (x, y)

    def horizontal_ratio(self):
        """Calculate horizontal gaze direction ratio.
        
        Returns a normalized value between 0.0 and 1.0 indicating
        horizontal gaze direction:
        - 0.0 = extreme right
        - 0.5 = center
        - 1.0 = extreme left
        
        Returns:
            float: Horizontal gaze ratio, or None if pupils not located
        """
        if self.pupils_located:
            # Calculate ratio for each eye (pupil position relative to eye center)
            # The formula normalizes pupil position within the eye region
            pupil_left = self.eye_left.pupil.x / (self.eye_left.center[0] * 2 - 10)
            pupil_right = self.eye_right.pupil.x / (self.eye_right.center[0] * 2 - 10)
            # Average both eyes for more stable results
            return (pupil_left + pupil_right) / 2

    def vertical_ratio(self):
        """Calculate vertical gaze direction ratio.
        
        Returns a normalized value between 0.0 and 1.0 indicating
        vertical gaze direction:
        - 0.0 = extreme top
        - 0.5 = center
        - 1.0 = extreme bottom
        
        Returns:
            float: Vertical gaze ratio, or None if pupils not located
        """
        if self.pupils_located:
            # Calculate ratio for each eye (pupil position relative to eye center)
            pupil_left = self.eye_left.pupil.y / (self.eye_left.center[1] * 2 - 10)
            pupil_right = self.eye_right.pupil.y / (self.eye_right.center[1] * 2 - 10)
            # Average both eyes for more stable results
            return (pupil_left + pupil_right) / 2

    def is_right(self):
        """Check if user is looking to the right.
        
        Returns:
            bool: True if looking right, False otherwise, or None if pupils not located
        """
        if self.pupils_located:
            # Threshold of 0.35 means pupil is positioned in the right portion of the eye
            return self.horizontal_ratio() <= 0.35

    def is_left(self):
        """Check if user is looking to the left.
        
        Returns:
            bool: True if looking left, False otherwise, or None if pupils not located
        """
        if self.pupils_located:
            # Threshold of 0.65 means pupil is positioned in the left portion of the eye
            return self.horizontal_ratio() >= 0.65

    def is_center(self):
        """Check if user is looking straight ahead (center).
        
        Returns:
            bool: True if looking center, False otherwise, or None if pupils not located
        """
        if self.pupils_located:
            # Center is defined as not clearly left or right
            return self.is_right() is not True and self.is_left() is not True

    def is_blinking(self):
        """Detect if the user is blinking (eyes closed).
        
        Uses the eye aspect ratio (width/height) to detect blinks.
        A high ratio (>3.8) indicates the eyes are closed.
        
        Returns:
            bool: True if blinking, False otherwise, or None if pupils not located
        """
        if self.pupils_located:
            # Calculate average blinking ratio from both eyes
            # Higher ratio = more closed (wider eye opening when closed)
            blinking_ratio = (self.eye_left.blinking + self.eye_right.blinking) / 2
            return blinking_ratio > 3.8

    def annotated_frame(self):
        """Get the current frame with pupil positions marked.
        
        Draws crosshairs (green lines) at the detected pupil locations
        to visualize where the system thinks the pupils are.
        
        Returns:
            numpy.ndarray: Annotated frame with pupil markers
        """
        frame = self.frame.copy()

        if self.pupils_located:
            # Green color for pupil markers
            color = (0, 255, 0)
            x_left, y_left = self.pupil_left_coords()
            x_right, y_right = self.pupil_right_coords()
            
            # Draw crosshairs at pupil locations (horizontal and vertical lines)
            cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
            cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
            cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
            cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)

        return frame
