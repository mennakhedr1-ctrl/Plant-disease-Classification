# Plant-disease-Classification
Image classification of plant leaf diseases using two approaches: a CNN built from scratch and a MobileNetV2 model fine-tuned with transfer learning.

## Overview
The project compares a CNN against a transfer learning model on the PlantVillage dataset. The MobileNetV2 model was trained in two stages: first only the new classification head, then the top layers fine-tuned with a low learning rate. Augmentation (rotation, zoom, shifts) was applied to the training data only.

## Approach
1. Loaded images at 224x224 with Keras 
2. CNN with 3 conv blocks and dense layers, trained from scratch for 10 epochs
3. **Transfer learning:** MobileNetV2 (ImageNet weights) with a pooling and dropout head
   - Stage 1: base frozen, head trained
   - Stage 2: last 30 layers unfrozen, Adam at 1e-5
4. Early stopping with best weights restored
5. Evaluated on the validation set with precision, recall, F1, and a normalized confusion matrix

6.## Results (validation set)
| Model | Validation accuracy | Macro F1 | Weighted F1 |
|-------|--------------------|----------|-------------|
| CNN (from scratch) | 88.87% | 0.827 | 0.884 |
| MobileNetV2 (transfer learning) | 96.7% | 0.957 | 0.966 |

## Tech Stack
Python, TensorFlow, Keras, Scikit-learn, NumPy, Matplotlib, Seaborn
