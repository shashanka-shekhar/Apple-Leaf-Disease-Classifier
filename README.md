# Apple Leaf Disease Classifier (MobileNetV2, 5 classes, TFLite)

An image classifier that identifies common apple leaf diseases and rejects non-leaf images. It is trained with transfer learning on MobileNetV2 and exported to TensorFlow Lite for mobile and on-device use.

**Classes:** `Apple Scab`, `Black Rot`, `Cedar Apple Rust`, `Healthy`, `Non Leaf`

**Result:** 99.49% accuracy on the held-out test set (196 images), 99.53% best validation accuracy.

## Highlights

- Combines four public datasets (PlantVillage, PlantDoc, Plant Pathology 2021, Leaf vs Non-Leaf) to mix lab-style and in-the-field photos.
- Includes a `Non Leaf` class so the app can reject irrelevant photos instead of forcing a disease label.
- Two-stage training: frozen backbone, then fine-tuning of the last 100 MobileNetV2 layers.
- Exported as a float TFLite model (about 8.5 MB, down from 25.9 MB for Keras) with a labels file.

## Data

| Source | Used for |
|---|---|
| PlantVillage (Apple, Train/Val/Test splits) | Train, validation, test |
| PlantDoc (train/test folders) | Train, validation (not used for Black Rot) |
| Plant Pathology 2021 – FGVC8 (960px) | Train only. Single-label images mapped: `scab`, `rust`, `frog_eye_leaf_spot` (as Black Rot), `healthy` |
| Leaf vs Non-Leaf images | `Non Leaf` class, split 85/15 into train/validation |

**Class balancing:** Raw class counts were uneven (Apple Scab 6,925; Healthy 6,711; Non Leaf 5,525; Black Rot 5,168; Cedar Apple Rust 1,856). Each class was randomly downsampled to the smallest class (1,856), giving **9,280 training images**. The validation set has **2,751 images** and is not balanced. The test set has **196 images** from the 4 disease/health classes.

## Model

- Backbone: MobileNetV2, ImageNet weights, input 224×224×3
- Preprocessing is inside the model (`Lambda(preprocess_input)`), so inputs are raw 0–255 RGB at inference
- Head: GlobalAveragePooling → Dropout(0.5) → Dense(5, softmax)
- Augmentation (train only): rotation 30°, shifts 20%, shear 20%, zoom 20%, horizontal and vertical flips, brightness 0.85–1.15
- Loss: sparse categorical cross-entropy, optimizer Adam (lr 1e-4)

## Training

1. **Frozen backbone, 10 epochs:** validation accuracy 95.20%, validation loss 0.2038.
2. **Fine-tuning, last 100 backbone layers unfrozen, up to 40 epochs:** EarlyStopping (patience 3, `val_loss`) stopped at epoch 11; the weights from epoch 8 were restored (val loss 0.0232, val accuracy 99.53%).

## Results (test set, 196 images)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Apple Scab | 1.00 | 0.98 | 0.99 | 51 |
| Black Rot | 1.00 | 1.00 | 1.00 | 50 |
| Cedar Apple Rust | 1.00 | 1.00 | 1.00 | 44 |
| Healthy | 0.98 | 1.00 | 0.99 | 51 |

Test accuracy 99.49%, test loss 0.0092. See [REPORT.md](REPORT.md) for analysis and limitations.

## Outputs

| File | Description |
|---|---|
| `Apple_Augmented_5_Classes.keras` | Full Keras model (~25.9 MB) |
| `Apple_Augmented_5_Classes.tflite` | Float TFLite model (~8.5 MB) |
| `Apple_Augmented_5_Classes_labels.txt` | Class names, one per line, in output-index order |

## Usage

Keras:

```python
import numpy as np, tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

model = tf.keras.models.load_model(
    "Apple_Augmented_5_Classes.keras",
    custom_objects={"preprocess_input": preprocess_input},
)
labels = open("Apple_Augmented_5_Classes_labels.txt").read().splitlines()

img = tf.keras.utils.load_img("leaf.jpg", target_size=(224, 224))
x = np.expand_dims(tf.keras.utils.img_to_array(img), 0)   # raw 0-255, no rescaling
pred = model.predict(x)[0]
print(labels[int(pred.argmax())], float(pred.max()))
```

TFLite:

```python
import numpy as np, tensorflow as tf

interp = tf.lite.Interpreter(model_path="Apple_Augmented_5_Classes.tflite")
interp.allocate_tensors()
inp, out = interp.get_input_details()[0], interp.get_output_details()[0]

img = tf.keras.utils.load_img("leaf.jpg", target_size=(224, 224))
x = np.expand_dims(tf.keras.utils.img_to_array(img), 0).astype("float32")
interp.set_tensor(inp["index"], x)
interp.invoke()
print(interp.get_tensor(out["index"])[0])
```

Note: the labels file is written from `sorted(class_indices.keys())`, which matches Keras' alphabetical class indexing.

## Reproducing

The work is in `plant-village.ipynb`, written for Kaggle with two T4 GPUs. Dataset paths under `/kaggle/input/...` must be adjusted to run elsewhere. Requires TensorFlow/Keras, pandas, scikit-learn, matplotlib and seaborn.

## Known limitations

- The test set is small (196 images) and does not include `Non Leaf`, so that class is only evaluated on validation data.
- Test images come from PlantVillage-style data, so the score is likely optimistic for real orchard photos.
- Only apple leaves and four conditions are covered.
