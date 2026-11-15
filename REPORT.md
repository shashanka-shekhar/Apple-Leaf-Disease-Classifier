# Report: Apple Leaf Disease Classification with MobileNetV2

## 1. Objective

Build a lightweight image classifier for apple leaves that distinguishes Apple Scab, Black Rot, Cedar Apple Rust and Healthy leaves, and also flags non-leaf images, so it can be deployed on mobile as a TFLite model.

## 2. Data

Four public sources were merged to mix controlled lab images with real-world field photos:

- **PlantVillage (Apple):** train, validation and test splits.
- **PlantDoc:** field images for Scab, Rust and Healthy (excluded for Black Rot by design). Its train/test folders feed train and validation.
- **Plant Pathology 2021 (FGVC8, 960px):** train only. Only images with a single exact label were used (`scab`, `rust`, `frog_eye_leaf_spot` mapped to Black Rot, `healthy`); multi-label images were not included.
- **Leaf vs Non-Leaf:** negative class, split 85/15 into train and validation.

### Class imbalance

| Class | Raw count |
| --- | --- |
| Apple Scab | 6,925 |
| Healthy | 6,711 |
| Non Leaf | 5,525 |
| Black Rot | 5,168 |
| Cedar Apple Rust | 1,856 |

All classes were randomly downsampled to 1,856 to give a balanced training set of 9,280 images. This keeps training unbiased across classes, at the cost of discarding about 18,000 images. Validation (2,751 images) was left unbalanced and unaugmented. The test set (196 images) contains four classes.

## 3. Method

**Architecture.** MobileNetV2 (ImageNet weights) at 224×224, with preprocessing embedded in the model, followed by global average pooling, 50% dropout and a 5-way softmax. Total 2.26M parameters; only 6,405 trainable in stage 1.

**Augmentation (train only).** Rotation ±30°, width/height shift 20%, shear 20%, zoom 20%, horizontal and vertical flips, brightness 0.85–1.15, nearest-fill.

**Training.**

- Stage 1: backbone frozen, Adam lr 1e-4, 10 epochs.
- Stage 2: last 100 backbone layers unfrozen, same learning rate, up to 40 epochs, EarlyStopping on validation loss (patience 3) with best-weights restore and a best-only checkpoint.

## 4. Results

### Training progression

| Stage | Epochs run | Best val loss | Val accuracy |
| --- | --- | --- | --- |
| Frozen backbone | 10 | 0.2038 | 95.20% |
| Fine-tuning | 11 (best = epoch 8) | 0.0232 | 99.53% |

Fine-tuning produced the main gain: validation loss fell from 0.20 to 0.02 within the first two epochs. Training accuracy in stage 1 (80.8%) was below validation accuracy (95.2%), which is expected given heavy augmentation and dropout on the training side only.

### Test performance (196 images)

- Accuracy **99.49%**, loss **0.0092**
- Macro-average precision, recall and F1 all about 1.00
- Black Rot and Cedar Apple Rust: perfect precision and recall
- Apple Scab recall 0.98 and Healthy precision 0.98: consistent with a single misclassification (one Scab image predicted as Healthy)

### Deployment

| Artifact | Size |
| --- | --- |
| Keras model | 25.86 MB |
| TFLite (float, builtin ops only) | 8.47 MB |

The TFLite export is about 3× smaller and uses only standard built-in operators, avoiding custom-op compatibility problems on mobile runtimes. No quantization was applied.

## 5. Discussion and limitations

1. **Optimistic test score.** The test set is small, so one error moves accuracy by about 0.5 points. Many PlantVillage images are lab-style shots with uniform backgrounds, and near-duplicate images across splits are a known issue in this dataset family. Real-world performance on orchard photos will likely be lower.
2. **`Non Leaf`** **is not tested.** The held-out test set only has four classes. The non-leaf class is evaluated only through the validation split.
3. **Validation role.** Validation data drove early stopping and checkpointing, so its 99.5% figure is slightly optimistic; the test figure is the cleaner estimate.
4. **Balancing trade-off.** Downsampling to 1,856 per class discards much of the Scab, Healthy, Non Leaf and Black Rot data. Class weights or oversampling Cedar Apple Rust could use more of the data.
5. **Label mapping.** Plant Pathology's `frog_eye_leaf_spot` was treated as Black Rot, and multi-label images were dropped. This is reasonable but should be stated when presenting results.
6. **Scope.** Four conditions on apple leaves only; fruit, stems and other diseases are out of scope. Predictions are not a substitute for expert diagnosis.

## 6. Suggested next steps

- Build a field-photo test set (phone photos, varied backgrounds) and report results separately for lab vs in-the-wild images.

## 7. Deliverables

`Apple_Augmented_5_Classes.keras`, `Apple_Augmented_5_Classes.tflite`, `Apple_Augmented_5_Classes_labels.txt`, and the training notebook.
