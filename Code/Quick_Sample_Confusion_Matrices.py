"""
FAST, small-sample confusion matrices for three (model, dataset) pairs -- meant for a
quick presentation check, not a substitute for the full formal evaluation. Uses only
N images per class (default 25) so it runs in under a couple minutes on CPU, instead
of the ~7 min per model a full pass over the whole toy dataset takes.

  1. June  (InHouse_0603) model  on  Merged_Collection_Museum_Toy  (4-class: Carabidae/Chrysomelidae/Other/Staphylinidae)
  2. July  (InHouse_0621) model  on  Merged_Collection_Museum_Toy
  3. Museum+Collection model (9-class) on InHouse_July/training -- its 6 non-InHouse
     classes (Cerambycidae, Corylophidae, Curculionidae, Ptilodactylidae, Scarabaeidae,
     Other) are summed into a single "Other" probability so it's directly comparable
     to the 4-class confusion matrices above.

Run with:
    C:\\Users\\timl9\\anaconda3\\python.exe Quick_Sample_Confusion_Matrices.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics import confusion_matrix

BASE = r"C:\Users\timl9\Downloads\Specimen_Classifier"
OUTPUT_DIR = os.path.join(BASE, "Presentation_Plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_PER_CLASS = 25
IMG_SIZE = (224, 224)
INHOUSE_CLASSES = ["Carabidae", "Chrysomelidae", "Other", "Staphylinidae"]  # alphabetical


def sample_and_predict(model, data_dir, classes):
    """Load N_PER_CLASS images per class subfolder, predict, return (true_idx, probs)."""
    true_idx = []
    all_probs = []
    for ci, cls in enumerate(classes):
        cls_dir = os.path.join(data_dir, cls)
        fnames = sorted(f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')))[:N_PER_CLASS]
        for fname in fnames:
            img = image.load_img(os.path.join(cls_dir, fname), target_size=IMG_SIZE)
            arr = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))
            probs = model.predict(arr, verbose=0)[0]
            all_probs.append(probs)
            true_idx.append(ci)
    return np.array(true_idx), np.array(all_probs)


def plot_cm(cm, labels, title, out_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(cm[i, j]), ha='center', va='center',
                     color='white' if cm[i, j] > cm.max() / 2 else 'black')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# ---- 1 & 2: June and July models on the toy data (both already 4-class) ----
TOY_DIR = os.path.join(BASE, "Data", "Merged_Collection_Museum_Toy")
MODELS_4CLASS = {
    "July": os.path.join(BASE, "Models", "InHouse_0621_ResNet50_1", "best_weights_ResNet50.h5"),
}

for label, weights_path in MODELS_4CLASS.items():
    print(f"\n[{label} on Toy] loading model...")
    model = tf.keras.models.load_model(weights_path)
    true_idx, probs = sample_and_predict(model, TOY_DIR, INHOUSE_CLASSES)
    pred_idx = np.argmax(probs, axis=1)
    acc = np.mean(pred_idx == true_idx)
    print(f"[{label} on Toy] sampled accuracy ({len(true_idx)} images): {acc:.3f}")
    cm = confusion_matrix(true_idx, pred_idx, labels=range(len(INHOUSE_CLASSES)))
    plot_cm(cm, INHOUSE_CLASSES, f'{label} model on Museum+Collection Toy (sample, n={len(true_idx)})',
            os.path.join(OUTPUT_DIR, f'QuickCM_{label}_on_Toy.png'))

# ---- 3: Museum+Collection (9-class) model on InHouse_July/training ----
JULY_TRAIN_DIR = os.path.join(BASE, "Data", "InHouse_July", "training")
MC_WEIGHTS = os.path.join(BASE, "Models", "Museum+Collection_ResNet50_1", "best_weights_ResNet50.h5")
MC_CLASSES = ["Carabidae", "Cerambycidae", "Chrysomelidae", "Corylophidae", "Curculionidae",
              "Other", "Ptilodactylidae", "Scarabaeidae", "Staphylinidae"]  # alphabetical, as trained

print("\n[Museum+Collection on July training] loading model...")
model = tf.keras.models.load_model(MC_WEIGHTS)
true_idx, probs = sample_and_predict(model, JULY_TRAIN_DIR, INHOUSE_CLASSES)

# Collapse the 9 output columns down to the 4 InHouse categories by summing
# probability mass for every class that isn't Carabidae/Chrysomelidae/Staphylinidae.
collapse_map = {c: (c if c in ("Carabidae", "Chrysomelidae", "Staphylinidae") else "Other") for c in MC_CLASSES}
collapsed_probs = np.zeros((probs.shape[0], len(INHOUSE_CLASSES)))
for j, mc_cls in enumerate(MC_CLASSES):
    target_idx = INHOUSE_CLASSES.index(collapse_map[mc_cls])
    collapsed_probs[:, target_idx] += probs[:, j]

pred_idx = np.argmax(collapsed_probs, axis=1)
acc = np.mean(pred_idx == true_idx)
print(f"[Museum+Collection on July training] sampled accuracy ({len(true_idx)} images, collapsed to 4 classes): {acc:.3f}")
cm = confusion_matrix(true_idx, pred_idx, labels=range(len(INHOUSE_CLASSES)))
plot_cm(cm, INHOUSE_CLASSES, f'Museum+Collection model on July training (sample, n={len(true_idx)}, collapsed to 4 classes)',
        os.path.join(OUTPUT_DIR, 'QuickCM_MuseumCollection_on_JulyTraining.png'))

print("\nDone -- all three quick confusion matrices saved to", OUTPUT_DIR)
