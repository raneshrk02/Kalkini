Violent Detection Project
This repository contains a project aimed at detecting violent activities in videos using deep learning techniques. The model leverages a combination of Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks to achieve this task.

Overview:
The goal of this project is to build a model that can detect violent activities in videos. This is accomplished by extracting features from video frames using a pre-trained MobileNetV2 model and then processing these features through a Bidirectional LSTM network for sequence learning.

Model Architecture

The model architecture consists of the following components:
- MobileNetV2: A pre-trained CNN used for feature extraction from video frames.
- Bidirectional LSTM: A recurrent neural network for capturing temporal dependencies in the sequence of frames.
- Dense Layers: Fully connected layers for classification.

Detailed Architecture

1. Feature Extraction:
   - MobileNetV2 model with pre-trained weights.
   - TimeDistributed wrapper to apply MobileNetV2 to each frame of the video.
   - GlobalAveragePooling1D layer to reduce the dimensionality.

2. Sequence Processing:
   - Bidirectional LSTM with dropout for sequence learning.
   - Dense layers with L2 regularization and dropout.

3. Classification:
   - Output layer with softmax activation for binary classification.

Dependencies:
- TensorFlow
- OpenCV
- NumPy
- scikit-learn
- Matplotlib
- Seaborn

The training script includes:
- Data preprocessing and augmentation.
- Splitting the data into training and testing sets.
- Defining the model architecture.
- Compiling the model with Adam optimizer and binary cross-entropy loss.
- Training the model with early stopping and learning rate reduction callbacks.
