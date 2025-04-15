import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
from collections import deque
from proto_model import ProtoNet
from utils import load_gesture_data
import mediapipe as mp

# Initialize Model and Load Trained Weights
model = ProtoNet()
model.load_state_dict(torch.load("proto_gesture.pth", map_location=torch.device('cpu')))
model.eval()

# Load support data and compute prototypes
data, labels, class_map = load_gesture_data(r"D:\College\Sign\proto\data1")
support_x = torch.tensor(data, dtype=torch.float32).reshape(-1, 50, 63)  # Ensure proper shape
support_y = torch.tensor(labels)

with torch.no_grad():
    emb_support = model(support_x)

def get_prototypes(support_x, support_y):
    prototypes = []
    for cls in torch.unique(support_y):
        class_feats = support_x[support_y == cls]
        proto = class_feats.mean(dim=0)
        prototypes.append(proto)
    return torch.stack(prototypes)

prototypes = get_prototypes(emb_support, support_y)
class_names = {v: k for k, v in class_map.items()}

# Mediapipe Hands Initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Extract and Normalize Landmarks
def extract_landmarks(results):
    if results.multi_hand_landmarks:
        coords = []
        for hand in results.multi_hand_landmarks:
            coords.extend([[p.x, p.y, p.z] for p in hand.landmark])
        coords = np.array(coords)
        center = coords[0]
        normalized = coords - center
        return normalized.flatten()
    return None

# Start Webcam
cap = cv2.VideoCapture(0)
sequence = deque(maxlen=50)

print("[INFO] Starting real-time recognition. Press 'q' to quit.")
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
        if landmarks is not None and landmarks.shape[0] == 63:
            sequence.append(landmarks)

    if len(sequence) == 50:
        seq_tensor = torch.tensor(np.array(sequence), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            emb = model(seq_tensor)
            dists = torch.cdist(emb, prototypes)
            pred = torch.argmin(dists, dim=1).item()
            gesture = class_names[pred]
            cv2.putText(frame, f"{gesture}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("Prototypical Gesture Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
