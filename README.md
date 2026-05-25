# Transfer Learning & Fine-Tuning for Computer Vision

This repository demonstrates the practical application of **Transfer Learning** and **Fine-Tuning** strategies to build highly accurate image classification models using limited training datasets. By leveraging pre-trained deep neural networks, these implementations significantly reduce computational overhead and training time while maintaining high performance.

---

## 🚀 Overview

Training deep learning models from scratch requires massive datasets and substantial computational power. This project showcases how to bypass these resource constraints by utilizing pre-trained architectures and adapting them to specific computer vision tasks.

The repository contains two primary implementations optimized for differing environment scales:
1. **Resource-Optimized Baseline:** A notebook structurally refactored to execute efficiently within resource-constrained environments (such as free-tier Google Colab instances) by managing memory consumption and optimizing data pipelines.
2. **Custom Domain Adaptation (Cats vs. Dogs):** An extension of the baseline architecture applied to a binary classification problem, demonstrating the end-to-end pipeline of data ingestion, feature extraction, and fine-tuning.

---

## 🛠️ Tech Stack & Architecture

* **Frameworks:** TensorFlow / Keras (or PyTorch, depending on your implementation)
* **Environment:** Google Colab / Jupyter Notebooks
* **Key Techniques:** Feature Extraction, Layer Freezing, Learning Rate Scheduling, Data Augmentation.

---

## 📁 Repository Structure

```text
├── images/
│   └── [Documentation assets]
├── transfer_learning_NeuralNetwork_DeepLearning_WORKING.ipynb       # Environment-optimized baseline notebook
└── transfer_learning_NeuralNetwork_DeepLearning_Cats_vs_Dogs.ipynb # Practical application on a binary image dataset
```

## 💻 Getting Started
Prerequisites
To run these notebooks locally or in the cloud, ensure you have the following packages installed:
``` bash
pip install tensorflow numpy matplotlib pillow
```

Execution via Google Colab
- Upload the target notebook (...WORKING.ipynb or ...Cats_vs_Dogs.ipynb) to your Google Drive.
- Open with Google Colab.
- Ensure the runtime type is configured to use a GPU accelerator (Runtime > Change runtime type > GPU) for optimal training performance.

## 📈 Key Methodology Highlights
* Memory Optimization: The codebase (https://colab.research.google.com/github/kylemath/ml4a-guides/blob/master/notebooks/transfer-learning.ipynb) specifically addresses common high-RAM bottlenecks during data loading by streaming images efficiently via optimized data generators rather than loading entire datasets into system memory at once.
* Layer Freezing Strategy: Initializes training by freezing the base convolutional layers to preserve learned low-level features (edges, textures), training only the custom dense classification head.
* Fine-Tuning Phase: Unfreezes top layers of the base network with a highly conservative learning rate to delicately align the model with the new target domain without destroying prior weights.

