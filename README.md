# ISL-Translator-DeepLearning
A deep learning-based translator for Indian Sign Language (ISL) that converts sign gestures into spoken sentences. Combines prototypical embedding, CNN-based gesture recognition, and sequence modeling to generate speech. Includes custom dataset, training pipeline, and real-time demo.

## 🔍 Overview

- **Goal:** Translate ISL sign language gestures to full sentences, and optionally convert them to audio.
- **Tech Stack:**
  - Python
  - TensorFlow / Keras
  - NumPy
  - OpenCV (if used)
- **Model:** CNN-based model for prototypic feature embedding.

## 🧠 Key Features

- Custom dataset for ISL gestures
- Preprocessing pipeline to clean and format data
- Prototype CNN model with evaluation
- End-to-end training and testing scripts
- Room for expansion to include sentence formation and speech generation

## 🚀 Getting Started

1. Clone the repo:

       git clone https://github.com/Shalmankursheeth/ISL-Translator-DeepLearning.git
       cd ISL-Translator-DeepLearning

2. Run training:

        python train.py

3. Test model:

        python proto_test.py

📊 Dataset
Currently includes a small set of 5 ISL words converted into .npy files. More data is being collected and labeled for sentence construction.

🔈 Future Work
Expand dataset with full sentence mappings

Implement real-time gesture capture

Add text-to-speech pipeline

✍️ Author
Mohamed Shalman Kursheeth K
NIT Puducherry | B.Tech CSE | Class of 2026
