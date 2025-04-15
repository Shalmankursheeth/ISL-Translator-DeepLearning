# pre.py
import cv2
import mediapipe as mp
import numpy as np
import os
from collections import deque

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Constants
SAVE_PATH = "data/yes"
GESTURE_NAME = input("Enter gesture label: ").strip()
os.makedirs(SAVE_PATH, exist_ok=True)

sequence = deque(maxlen=50)
cap = cv2.VideoCapture(0)

def extract_landmarks(results):
    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0]
        coords = np.array([[p.x, p.y, p.z] for p in lm.landmark])
        center = coords[0]
        normalized = coords - center  # Normalize
        return normalized.flatten()
    return None

print("[INFO] Starting capture. Show the gesture...")
print("[INFO] Press 's' to save when 50 frames are recorded.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        landmarks = extract_landmarks(results)
        if landmarks is not None:
            sequence.append(landmarks)

    if len(sequence) == 50:
        cv2.putText(frame, "Ready to Save [Press S]", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow("Recording Gesture", frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('s') and len(sequence) == 50:
        np.save(os.path.join(SAVE_PATH, f"{GESTURE_NAME}.npy"), np.array(sequence))
        print("[INFO] Saved gesture:", GESTURE_NAME)
        break
    elif key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
