# MNIST-Digit-Classification
This project implements three machine learning/deep learning models to classify handwritten digits from the MNIST dataset: - Artificial Neural Network (ANN) - Support Vector Machine (SVM) - Convolutional Neural Network (CNN)
# MNIST Digit Classification: ANN, SVM, and CNN Comparison

This project implements three machine learning/deep learning models to classify handwritten digits from the MNIST dataset:
- Artificial Neural Network (ANN)
- Support Vector Machine (SVM)
- Convolutional Neural Network (CNN)

---

## Project Overview

The MNIST dataset contains 70,000 grayscale images of handwritten digits (28x28 pixels) split into training and test sets. This project covers:

- Data loading, preprocessing, and normalization
- Model implementations:
  - ANN with fully connected layers
  - SVM with RBF kernel
  - CNN with configurable hyperparameters
- Training and evaluation of models
- Experimentation with different hyperparameters:
  - Learning rates
  - Number of fully connected layers
  - Batch sizes
  - Activation functions
  - Optimizers
  - Dropout rates

---

## Dependencies

- Python 3.x
- TensorFlow / Keras
- scikit-learn
- NumPy
- Matplotlib

Install required packages with:
```bash
pip install tensorflow scikit-learn numpy matplotlib
