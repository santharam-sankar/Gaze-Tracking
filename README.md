# GazeTracking

A Python library for real-time gaze tracking using computer vision. This library can detect eye movements, track pupil positions, and identify when a user is blinking or looking in different directions (left, right, center).

## Features

- 👁️ **Real-time gaze tracking** - Track eye movements in real-time using webcam
- 🎯 **Pupil detection** - Accurately detect and track pupil positions
- 👀 **Blink detection** - Identify when eyes are closed or blinking
- 📍 **Direction detection** - Determine if user is looking left, right, or center
- 🎨 **Visual annotations** - Annotated frames with pupil markers

## Requirements

- Python 3.6+
- OpenCV 3.4+
- dlib 19.17+
- NumPy 1.16+
- Webcam access

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sudheerakotha/GazeTracking.git
cd GazeTracking
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the facial landmarks model:
The project includes a pre-trained model file (`shape_predictor_68_face_landmarks.dat`) in the `gaze_tracking/trained_models/` directory. If it's missing, you can download it from [dlib's website](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2).

## Quick Start

Simply run the main script:
```bash
python main.py
```

The script will:
- Access your webcam
- Track your gaze in real-time
- Display annotations showing where you're looking
- Show pupil coordinates
- Press ESC to exit

### Using as a Library

You can also import and use the GazeTracking class in your own code:

```python
import cv2
from gaze_tracking import GazeTracking

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

while True:
    _, frame = webcam.read()
    gaze.refresh(frame)
    
    frame = gaze.annotated_frame()
    text = ""
    
    if gaze.is_blinking():
        text = "Blinking"
    elif gaze.is_right():
        text = "Looking right"
    elif gaze.is_left():
        text = "Looking left"
    elif gaze.is_center():
        text = "Looking center"
    
    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
    
    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    
    cv2.imshow("Demo", frame)
    
    if cv2.waitKey(1) == 27:  # Press ESC to exit
        break

webcam.release()
cv2.destroyAllWindows()
```

## API Reference

### GazeTracking Class

#### Methods

- `refresh(frame)` - Refreshes the frame and analyzes it
  - **Parameters**: `frame` (numpy.ndarray) - The frame to analyze
  
- `pupil_left_coords()` - Returns the coordinates of the left pupil
  - **Returns**: `(x, y)` tuple or `None`
  
- `pupil_right_coords()` - Returns the coordinates of the right pupil
  - **Returns**: `(x, y)` tuple or `None`
  
- `horizontal_ratio()` - Returns a number between 0.0 and 1.0 indicating horizontal gaze direction
  - **Returns**: `float` - 0.0 (extreme right), 0.5 (center), 1.0 (extreme left)
  
- `vertical_ratio()` - Returns a number between 0.0 and 1.0 indicating vertical gaze direction
  - **Returns**: `float` - 0.0 (extreme top), 0.5 (center), 1.0 (extreme bottom)
  
- `is_right()` - Returns `True` if the user is looking to the right
  - **Returns**: `bool`
  
- `is_left()` - Returns `True` if the user is looking to the left
  - **Returns**: `bool`
  
- `is_center()` - Returns `True` if the user is looking to the center
  - **Returns**: `bool`
  
- `is_blinking()` - Returns `True` if the user is blinking
  - **Returns**: `bool`
  
- `annotated_frame()` - Returns the frame with pupils highlighted
  - **Returns**: `numpy.ndarray` - Annotated frame

#### Properties

- `pupils_located` - `True` if pupils have been successfully located, `False` otherwise

## Project Structure

```
GazeTracking/
├── gaze_tracking/
│   ├── __init__.py
│   ├── gaze_tracking.py    # Main GazeTracking class
│   ├── eye.py              # Eye detection and isolation
│   ├── pupil.py            # Pupil detection
│   ├── calibration.py      # Calibration utilities
│   └── trained_models/
│       └── shape_predictor_68_face_landmarks.dat
├── main.py                 # Entry point - run this to start
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Troubleshooting

### Webcam not working
- Ensure your webcam is properly connected and accessible
- On Linux, you may need to grant camera permissions
- Try changing the camera index in `cv2.VideoCapture(0)` to `1` or `2`

### dlib installation issues
- On Windows, you may need to install Visual Studio Build Tools
- On Linux/Mac, ensure you have cmake installed: `sudo apt-get install cmake`
- For easier installation, try: `pip install dlib` (may require pre-built wheels)

### Model file missing
- Download `shape_predictor_68_face_landmarks.dat` from [dlib's website](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
- Extract and place it in `gaze_tracking/trained_models/` directory

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Uses [dlib](http://dlib.net/) for facial landmark detection
- Uses [OpenCV](https://opencv.org/) for computer vision operations
- Facial landmarks model from dlib's trained models

## Author

Created for real-time gaze tracking applications.

