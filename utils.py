import os
import numpy as np
from sklearn.model_selection import train_test_split

def load_gesture_data(root_dir):
    data, labels = [], []
    class_map = {}
    idx = 0

    for gesture in os.listdir(root_dir):
        gesture_path = os.path.join(root_dir, gesture)
        if not os.path.isdir(gesture_path):
            continue
        class_map[gesture] = idx
        for file in os.listdir(gesture_path):
            if file.endswith(".npy"):
                seq = np.load(os.path.join(gesture_path, file))
                if seq.shape == (50, 63):  # Validate shape
                    data.append(seq)
                    labels.append(idx)
        idx += 1

    data, labels = np.array(data), np.array(labels)
    return data, labels, class_map
