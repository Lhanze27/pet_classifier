"""
==========================================================================
MODEL EVALUATION SCRIPT - Generates thesis-ready figures and metrics
==========================================================================
Run this AFTER 01_train_model.py completes.

HOW TO RUN:
    python 02_evaluate_model.py

OUTPUTS:
    - outputs/figures/confusion_matrix.png
    - outputs/figures/sample_predictions.png
    - outputs/figures/per_class_accuracy.png
    - outputs/figures/gradcam_samples.png
    - outputs/classification_report.txt
==========================================================================
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

# ==========================================================================
# GPU SETUP
# ==========================================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# ==========================================================================
# LOAD MODEL + CLASS NAMES
# ==========================================================================
print("Loading trained model...")
model = keras.models.load_model('outputs/models/final_model.keras')

with open('outputs/class_names.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

print(f"✅ Model loaded with {len(class_names)} classes")

# ==========================================================================
# LOAD TEST DATA
# ==========================================================================
IMG_SIZE = 224
BATCH_SIZE = 16

def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32)
    return image, label

test_ds = tfds.load('oxford_iiit_pet', split='test', as_supervised=True)
test_ds_batched = (test_ds
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE))

# ==========================================================================
# GET PREDICTIONS
# ==========================================================================
print("Generating predictions on test set...")
y_true = []
y_pred = []

for images, labels in test_ds_batched:
    predictions = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(predictions, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# ==========================================================================
# CONFUSION MATRIX
# ==========================================================================
print("Building confusion matrix...")
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(20, 18))
sns.heatmap(cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names,
    cbar_kws={'label': 'Number of predictions'})
plt.title('Confusion Matrix - Pet Breed Classification', fontsize=16)
plt.xlabel('Predicted Breed', fontsize=12)
plt.ylabel('True Breed', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('outputs/figures/confusion_matrix.png', dpi=200, bbox_inches='tight')
plt.close()
print("✅ Confusion matrix saved")

# ==========================================================================
# CLASSIFICATION REPORT (precision, recall, F1 per class)
# ==========================================================================
print("Generating classification report...")
report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print(report)

with open('outputs/classification_report.txt', 'w') as f:
    f.write("CLASSIFICATION REPORT\n")
    f.write("=" * 70 + "\n\n")
    f.write(report)
print("✅ Classification report saved")

# ==========================================================================
# SAMPLE PREDICTIONS VISUALIZATION (correct + incorrect examples)
# ==========================================================================
print("Creating sample predictions figure...")

# Get some test images with predictions
sample_images = []
sample_true = []
sample_pred = []
sample_conf = []

for images, labels in test_ds_batched.take(2):
    preds = model.predict(images, verbose=0)
    for i in range(len(images)):
        sample_images.append(images[i].numpy())
        sample_true.append(labels[i].numpy())
        sample_pred.append(np.argmax(preds[i]))
        sample_conf.append(np.max(preds[i]))

# Plot 16 samples (4x4 grid)
fig, axes = plt.subplots(4, 4, figsize=(16, 16))
for idx, ax in enumerate(axes.flat):
    if idx < len(sample_images):
        img = sample_images[idx].astype('uint8')
        true_label = class_names[sample_true[idx]]
        pred_label = class_names[sample_pred[idx]]
        confidence = sample_conf[idx]

        ax.imshow(img)
        color = 'green' if sample_true[idx] == sample_pred[idx] else 'red'
        ax.set_title(
            f"True: {true_label}\nPred: {pred_label} ({confidence*100:.1f}%)",
            color=color, fontsize=9
        )
        ax.axis('off')

plt.suptitle('Sample Predictions (Green = Correct, Red = Wrong)', fontsize=14)
plt.tight_layout()
plt.savefig('outputs/figures/sample_predictions.png', dpi=200, bbox_inches='tight')
plt.close()
print("✅ Sample predictions figure saved")

# ==========================================================================
# PER-CLASS ACCURACY BAR CHART
# ==========================================================================
print("Creating per-class accuracy chart...")
per_class_acc = cm.diagonal() / cm.sum(axis=1)

plt.figure(figsize=(14, 10))
sorted_indices = np.argsort(per_class_acc)
sorted_names = [class_names[i] for i in sorted_indices]
sorted_accs = per_class_acc[sorted_indices]

colors = ['#ff6b6b' if acc < 0.8 else '#51cf66' for acc in sorted_accs]
plt.barh(range(len(sorted_names)), sorted_accs, color=colors)
plt.yticks(range(len(sorted_names)), sorted_names)
plt.xlabel('Accuracy')
plt.title('Per-Class Accuracy (Red = Below 80%)')
plt.axvline(x=0.8, color='black', linestyle='--', alpha=0.5)
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig('outputs/figures/per_class_accuracy.png', dpi=200, bbox_inches='tight')
plt.close()
print("✅ Per-class accuracy chart saved")

# ==========================================================================
# GRAD-CAM VISUALIZATIONS (model interpretability)
# ==========================================================================
# Grad-CAM (Selvaraju et al., 2017) highlights the image regions that most
# influenced the model's predicted class. Useful for the thesis as a
# "where the model looks" interpretability figure.
print("Creating Grad-CAM figure...")

def _find_last_conv_layer(m):
    # Find the EfficientNetV2B0 sub-model embedded in our trained model.
    for layer in m.layers:
        if isinstance(layer, keras.Model):
            return layer
    return None

BASE_MODEL = _find_last_conv_layer(model)
if BASE_MODEL is None:
    raise RuntimeError("Could not locate base sub-model for Grad-CAM.")

# Head = layers AFTER the base sub-model (GAP -> Dropout -> Dense -> Dropout -> Dense).
_seen, HEAD_LAYERS = False, []
for _layer in model.layers:
    if _layer is BASE_MODEL:
        _seen = True
        continue
    if _seen:
        HEAD_LAYERS.append(_layer)

def compute_gradcam(model, img_batch, class_idx):
    # Single tape, single forward pass — features and preds share the graph,
    # so gradients flow from class score back to the conv feature map.
    with tf.GradientTape() as tape:
        features = BASE_MODEL(img_batch, training=False)
        tape.watch(features)
        x = features
        for layer in HEAD_LAYERS:
            x = layer(x, training=False)
        loss = x[:, class_idx]
    grads = tape.gradient(loss, features)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = features[0]
    heatmap = conv_out @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

# Pick 8 test images (mix of correct + incorrect if available).
gc_images, gc_true, gc_pred, gc_conf = [], [], [], []
for images, labels in test_ds_batched.take(2):
    preds = model.predict(images, verbose=0)
    for i in range(len(images)):
        gc_images.append(images[i].numpy())
        gc_true.append(int(labels[i].numpy()))
        gc_pred.append(int(np.argmax(preds[i])))
        gc_conf.append(float(np.max(preds[i])))
        if len(gc_images) >= 8:
            break
    if len(gc_images) >= 8:
        break

fig, axes = plt.subplots(2, 8, figsize=(20, 6))
for col in range(8):
    img = gc_images[col]
    img_batch = tf.expand_dims(tf.constant(img, dtype=tf.float32), 0)
    heat = compute_gradcam(model, img_batch, gc_pred[col])
    # upsample heatmap to image size
    heat_resized = tf.image.resize(heat[..., tf.newaxis], (IMG_SIZE, IMG_SIZE)).numpy().squeeze()

    img_u8 = img.astype('uint8')
    correct = gc_true[col] == gc_pred[col]
    title = (f"True: {class_names[gc_true[col]]}\n"
             f"Pred: {class_names[gc_pred[col]]} ({gc_conf[col]*100:.0f}%)")
    color = 'green' if correct else 'red'

    axes[0, col].imshow(img_u8)
    axes[0, col].set_title(title, color=color, fontsize=8)
    axes[0, col].axis('off')

    axes[1, col].imshow(img_u8)
    axes[1, col].imshow(heat_resized, cmap='jet', alpha=0.45)
    axes[1, col].axis('off')

axes[0, 0].set_ylabel('Input', fontsize=11)
axes[1, 0].set_ylabel('Grad-CAM', fontsize=11)
plt.suptitle('Grad-CAM: where the model looks when predicting the breed', fontsize=13)
plt.tight_layout()
plt.savefig('outputs/figures/gradcam_samples.png', dpi=200, bbox_inches='tight')
plt.close()
print("✅ Grad-CAM figure saved")

# ==========================================================================
# SUMMARY
# ==========================================================================
overall_acc = np.mean(y_true == y_pred)
print("\n" + "=" * 70)
print(f"🎯 OVERALL ACCURACY: {overall_acc * 100:.2f}%")
print(f"🎯 BEST PERFORMING CLASS: {class_names[np.argmax(per_class_acc)]} ({per_class_acc.max()*100:.1f}%)")
print(f"🎯 WORST PERFORMING CLASS: {class_names[np.argmin(per_class_acc)]} ({per_class_acc.min()*100:.1f}%)")
print("=" * 70)
print("\n✅ ALL THESIS FIGURES READY in outputs/figures/")
print("✅ Check outputs/classification_report.txt for detailed metrics")
