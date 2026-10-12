Apple Leaf Disease Classifier

An end-to-end Computer Vision project designed to detect and classify foliar diseases in apple trees from leaf imagery. This tool aims to assist farmers, agronomists, and researchers in early diagnosis and targeted intervention, minimizing crop yield loss.

---

Table of Contents

- [Overview](#-overview)
- [Supported Classes](#-supported-classes)
- [Dataset](#-dataset)
- [Model Architecture & Performance](#-model-architecture--performance)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
  - [1. Training](#1-training)
  - [2. Evaluation & Inference](#2-evaluation--inference)
  - [3. Interactive Web App (Optional)](#3-interactive-web-app-optional)
- [Results & Visualizations](#-results--visualizations)
- [Contributing](#-contributing)
- [License](#-license)

---

Overview

Apple orchards are susceptible to several foliar diseases that drastically reduce fruit quality and production volume. This repository provides:
- Data preprocessing and augmentation pipelines for plant pathology images.
- Deep learning classification models (e.g., Transfer Learning using MobileNetV2, EfficientNet, or ResNet).
- Evaluation scripts reporting metrics like Accuracy, Precision, Recall, and F1-score.
- Inference tools for rapid single-image and batch classification.

---

Supported Classes

The classifier typically distinguishes among the following leaf conditions (customizable based on your dataset):

| Class | Description |
| :--- | :--- |
| **Healthy** | Foliage showing no signs of visible disease or fungal infection. |
| **Apple Scab** | Fungal infection caused by *Venturia inaequalis*, showing olive-green to black velvety spots. |
| **Black Rot** | Fungal infection caused by *Diplodia seriata*, presenting circular brown lesions and frog-eye leaf spots. |
| **Cedar Apple Rust** | Fungal disease caused by *Gymnosporangium juniperi-virginianae*, displaying bright orange-yellow lesions. |

---

Dataset

This project is configured to work with public agricultural benchmark datasets, such as:
- **[PlantVillage Dataset](https://www.kaggle.com/datasets/tushar5harma/plant-village-dataset-updated)**
- **[Plant Pathology Challenge (FGVC / Kaggle)](https://www.kaggle.com/datasets/jirkaborovec/plant-pathology-2021-fgvc8-960px)**
- [PlantDoc](https://www.kaggle.com/datasets/nirmalsankalana/plantdoc-dataset)



Directory Setup
Ensure your data follows a standard image classification directory format:

```text
data/
├── train/
│   ├── healthy/
│   ├── apple_scab/
│   ├── black_rot/
│   └── cedar_apple_rust/
├── val/
│   ├── healthy/
│   └── ...
└── test/
    ├── healthy/
    └── ...
```

---

Model Architecture & Performance

- **Backbone Architecture:** Transfer learning with MobileNetV2** (pre-trained on ImageNet).
- **Optimization:** Adam optimizer with categorical cross-entropy loss and learning rate decay.
  
- **Evaluation Metrics:**

| Model | Accuracy | F1-Score | Inference Time (CPU) |
| :--- | :---: | :---: | :---: |
| MobileNetV2 | ~96.5% | 0.96 | ~76.5 ms |
---

Results & Visualizations

- **Confusion Matrix:** View misclassifications across similar fungal patterns.
- **Grad-CAM Saliency Maps:** Visual heatmaps highlighting the exact regions of the leaf the neural network attends to when assigning disease labels.



License

Distributed under the MIT License. See `LICENSE` for more information.
