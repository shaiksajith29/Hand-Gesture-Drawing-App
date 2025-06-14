import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Canvas setup
width, height = 1280, 960
display_width, display_height = 640, 480
canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
draw_color = (0, 0, 255)  # Default color
eraser_color = (255, 255, 255)
brush_thickness = 5
eraser_thickness = 20

# Variables
canvas_x, canvas_y = 0, 0
prev_x, prev_y = None, None
drawing = erasing = panning = False

# Color buttons with images
button_size = 50
color_buttons = [
    {"color": (0, 0, 255), "pos": (10, 10), "image": cv2.imread("red.png")},
    {"color": (0, 255, 0), "pos": (70, 10), "image": cv2.imread("green.png")},
    {"color": (255, 0, 0), "pos": (130, 10), "image": cv2.imread("blue.png")},
    {"color": (0, 0, 0), "pos": (190, 10), "image": cv2.imread("black.png")},
]

# Check image loading
for btn in color_buttons:
    if btn["image"] is None:
        print(f"Warning: Image for color {btn['color']} not found.")

def is_finger_extended(tip, pip, wrist, threshold=0.1):
    tip_to_wrist = np.linalg.norm([tip.x - wrist.x, tip.y - wrist.y])
    pip_to_wrist = np.linalg.norm([pip.x - wrist.x, pip.y - wrist.y])
    return tip_to_wrist > pip_to_wrist + threshold

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # Draw image buttons
    for btn in color_buttons:
        x, y = btn["pos"]
        if btn["image"] is not None:
            resized = cv2.resize(btn["image"], (button_size, button_size))
            frame[y:y + button_size, x:x + button_size] = resized
        else:
            cv2.rectangle(frame, (x, y), (x + button_size, y + button_size), btn["color"], -1)
        cv2.rectangle(frame, (x, y), (x + button_size, y + button_size), (255, 255, 255), 2)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            lm = hand_landmarks.landmark
            index_tip, index_pip = lm[8], lm[6]
            middle_tip, middle_pip = lm[12], lm[10]
            ring_tip, ring_pip = lm[16], lm[14]
            pinky_tip, pinky_pip = lm[20], lm[18]
            wrist = lm[0]

            x_index = int(index_tip.x * display_width)
            y_index = int(index_tip.y * display_height)

            # Handle color selection
            if y_index < button_size + 10:
                for btn in color_buttons:
                    bx, by = btn["pos"]
                    if bx < x_index < bx + button_size and by < y_index < by + button_size:
                        draw_color = btn["color"]
                        print(f"Color changed to {draw_color}")

            # Finger states
            index_extended = is_finger_extended(index_tip, index_pip, wrist)
            middle_extended = is_finger_extended(middle_tip, middle_pip, wrist)
            ring_extended = is_finger_extended(ring_tip, ring_pip, wrist)
            pinky_extended = is_finger_extended(pinky_tip, pinky_pip, wrist)

            extended_fingers = sum([index_extended, middle_extended, ring_extended, pinky_extended])
            dist_im = np.linalg.norm([index_tip.x - middle_tip.x, index_tip.y - middle_tip.y]) * display_width

            # Determine mode
            if extended_fingers >= 3:
                erasing, drawing, panning = True, False, False
            elif index_extended and middle_extended and dist_im < 40:
                panning, drawing, erasing = True, False, False
            elif index_extended and not middle_extended:
                drawing, erasing, panning = True, False, False
            else:
                drawing = erasing = panning = False

            canvas_x_index = x_index + canvas_x
            canvas_y_index = y_index + canvas_y

            if panning and prev_x is not None and prev_y is not None:
                canvas_x -= (x_index - prev_x)
                canvas_y -= (y_index - prev_y)
                canvas_x = max(0, min(canvas_x, width - display_width))
                canvas_y = max(0, min(canvas_y, height - display_height))
            elif drawing and prev_x is not None and prev_y is not None:
                cv2.line(canvas, (prev_x + canvas_x, prev_y + canvas_y),
                         (canvas_x_index, canvas_y_index), draw_color, brush_thickness)
            elif erasing and prev_x is not None and prev_y is not None:
                cv2.line(canvas, (prev_x + canvas_x, prev_y + canvas_y),
                         (canvas_x_index, canvas_y_index), eraser_color, eraser_thickness)

            prev_x, prev_y = x_index, y_index
            cv2.circle(frame, (x_index, y_index), 5, (0, 255, 0), -1)
    else:
        prev_x, prev_y = None, None

    # Crop canvas
    canvas_view = canvas[canvas_y:canvas_y + display_height, canvas_x:canvas_x + display_width]
    if canvas_view.shape[:2] != (display_height, display_width):
        canvas_view = np.ones((display_height, display_width, 3), dtype=np.uint8) * 255

    combined = cv2.addWeighted(frame, 0.5, canvas_view, 0.5, 0)
    mode = "Erasing" if erasing else "Panning" if panning else "Drawing" if drawing else "Idle"
    cv2.putText(combined, f"Mode: {mode}", (10, display_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Tracking Drawing", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"canvas_{timestamp}.png"
        cv2.imwrite(filename, canvas)
        print(f"Saved as {filename}")
    elif key == ord('c'):
        canvas[:] = 255
        print("Canvas cleared.")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
