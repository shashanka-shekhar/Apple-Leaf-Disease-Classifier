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
- **[PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)**
- **[Plant Pathology Challenge (FGVC / Kaggle)](https://www.kaggle.com/datasets/jirkaborovec/plant-pathology-2021-fgvc8-960px)**
- [PlantDoc](https://www.kaggle.com/datasets/nirmalsankalana/plantdoc-dataset)**



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

- **Backbone Architecture:** Transfer learning with **ResNet-50 / EfficientNet-B0 / MobileNetV2** (pre-trained on ImageNet).
- **Optimization:** Adam optimizer with categorical cross-entropy loss and learning rate decay.
- **Evaluation Metrics:**

| Model | Accuracy | F1-Score | Inference Time (CPU) |
| :--- | :---: | :---: | :---: |
| MobileNetV2 | ~96.5% | 0.96 | ~35 ms |
| ResNet50 | ~98.1% | 0.98 | ~85 ms |

*(Update the table above with your specific model benchmarks.)*

---

Project Structure

```text
apple-disease-classifier/
├── data/                      # Dataset directory (ignored by git)
├── models/                    # Saved model checkpoints and weights
├── notebooks/                 # Exploratory Data Analysis & experiments
├── src/
│   ├── __init__.py
│   ├── dataset.py             # Dataloaders and data augmentations
│   ├── model.py               # Model definition and transfer learning setup
│   ├── train.py               # Training loop and validation
│   └── predict.py             # Single/batch image inference script
├── app.py                     # Optional Streamlit/Gradio deployment app
├── requirements.txt           # Python dependencies
├── .gitignore
└── README.md
```

---

Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/apple-disease-classifier.git
   cd apple-disease-classifier
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate    # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

Usage

### 1. Training
To train the model from scratch or fine-tune a pre-trained backbone:
```bash
python src/train.py --epochs 25 --batch-size 32 --lr 1e-4 --backbone resnet50
```

### 2. Evaluation & Inference
Run inference on a single test image:
```bash
python src/predict.py --image path/to/leaf_image.jpg --weights models/best_model.pth
```

**Example output:**
```json
{
  "prediction": "Apple Scab",
  "confidence": 0.974
}
```

### 3. Interactive Web App (Optional)
If you are using Streamlit or Gradio for a browser demo:
```bash
streamlit run app.py
```

---

Results & Visualizations

- **Confusion Matrix:** View misclassifications across similar fungal patterns.
- **Grad-CAM Saliency Maps:** Visual heatmaps highlighting the exact regions of the leaf the neural network attends to when assigning disease labels.

*(Add your plots or screenshots here inside an `assets/` directory, e.g., `![Confusion Matrix](assets/confusion_matrix.png)`)*

---

Contributing

Contributions, bug reports, and feature requests are welcome!
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/NewFeature`)
3. Commit your Changes (`git commit -m 'Add NewFeature'`)
4. Push to the Branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

---

License

Distributed under the MIT License. See `LICENSE` for more information.
