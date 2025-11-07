"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
import sys
from gaze_tracking import GazeTracking

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Unable to open webcam")
    sys.exit(1)

try:
    while True:
        ok, frame = webcam.read()
        if not ok or frame is None:
            print("Failed to grab frame")
            break

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

        cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (0, 0, 0), 2)

        # Clean output – remove np.int32 wrapper
        left_pupil = gaze.pupil_left_coords()
        right_pupil = gaze.pupil_right_coords()

        if left_pupil is not None:
            left_pupil = (int(left_pupil[0]), int(left_pupil[1]))
        if right_pupil is not None:
            right_pupil = (int(right_pupil[0]), int(right_pupil[1]))

        cv2.putText(frame, f"Left pupil:  {left_pupil}", (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 0), 2)
        cv2.putText(frame, f"Right pupil: {right_pupil}", (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 0), 2)

        cv2.imshow("Demo", frame)

        # Exit on ESC or 'q'
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            break

except KeyboardInterrupt:
    # Graceful handling of Ctrl+C
    pass
except Exception as e:
    print("Exception:", e)
finally:
    if webcam is not None and webcam.isOpened():
        webcam.release()
    cv2.destroyAllWindows()
    print("Clean exit.")
