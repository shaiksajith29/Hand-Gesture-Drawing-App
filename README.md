# Hand Gesture Drawing App using OpenCV and MediaPipe

This project is a gesture-controlled drawing application that uses computer vision techniques to detect hand landmarks and allow the user to draw, erase, pan, and select colors using only their fingers and webcam input.

## Features

- Draw on a digital canvas using index finger tracking.
- Select colors using hand gestures over visual buttons.
- Erase with three or more fingers extended.
- Pan across a large canvas using two fingers close together.
- Save the canvas with the 's' key.
- Clear the canvas with the 'c' key.
- Quit the application with the 'q' key.

## Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy
- Webcam input for real-time interaction

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- MediaPipe
- NumPy

Install dependencies:
```bash
pip install opencv-python mediapipe numpy
