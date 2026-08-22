"""
Configurable model-comparison script: predicts with N models on one dataset and plots
per-class ROC/PR curves (for a chosen TARGET_CLASS), a "cumulative" ROC/PR curve per
model (micro- or macro-averaged across every eval class), and a confusion matrix per
model.

TO COMPARE SOMETHING DIFFERENT: edit the CONFIG block below only -- change DATASET_DIR,
EVAL_CLASSES, MODELS, TARGET_CLASS, or AVERAGING. Nothing past the CONFIG block needs
to change.

A model whose own output classes don't exactly match EVAL_CLASSES (e.g. a 9-class
Museum+Collection model being scored against a 4-class InHouse scheme) can supply a
"mapping" that sums its extra classes' probability into one of EVAL_CLASSES (usually
"Other"). Likewise, if DATASET_DIR's subfolders are more fine-grained than EVAL_CLASSES,
set DATASET_LABEL_MAPPING the same way to collapse ground-truth labels.

Predictions are cached to disk under Presentation_Plots/cache/, keyed by a hash of
(model, dataset, eval classes, mapping) -- so changing any of those in CONFIG
automatically invalidates the old cache instead of silently reusing stale results.

Run with:
    C:\\Users\\timl9\\anaconda3\\python.exe Compare_Models_on_Dataset.py
"""

import os
import re
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless: save figures only, never try to open a display window
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix

# =========================================== CONFIG ===========================================
BASE = r"C:\Users\timl9\Downloads\Specimen_Classifier"

# Dataset to evaluate on. Must be a flat folder of class-named subfolders
# (folder/ClassName/*.jpg), no train/test split.
DATASET_DIR = os.path.join(BASE, "Data", "Merged_Collection_Museum_Toy")

# Ground-truth class scheme everything gets scored against.
EVAL_CLASSES = ["Carabidae", "Chrysomelidae", "Other", "Staphylinidae"]

# Only needed if DATASET_DIR's subfolders are more fine-grained than EVAL_CLASSES,
# e.g. {"Curculionidae": "Other", "Scarabaeidae": "Other"}. Any subfolder name not
# listed here is assumed to already match an entry in EVAL_CLASSES exactly.
DATASET_LABEL_MAPPING = {}

# Models to compare -- add/remove/edit entries freely.
#   weights : path to the .h5 file
#   classes : the model's own output class order, as trained (Keras sorts subfolder
#             names alphabetically, so this is just the alphabetically-sorted list of
#             class folders the model was trained on)
#   mapping : only needed if `classes` isn't identical to EVAL_CLASSES -- maps each
#             extra model class to one of EVAL_CLASSES. Omit (or {}) if they match.
MODELS = {
    "InHouse_June": {
        "weights": os.path.join(BASE, "Models", "InHouse_0603_ResNet50_1", "best_weights_ResNet50.h5"),
        "classes": ["Carabidae", "Chrysomelidae",
                    "Other", "Staphylinidae"],
        "mapping": {}
    },
    "InHouse_July": {"weights": os.path.join(BASE, "Models", "InHouse_0621_ResNet50_1", "best_weights_ResNet50.h5"),
            "classes": ["Carabidae", "Chrysomelidae",
                        "Other", "Staphylinidae"],
            "mapping": {}
    }
}
    # Example of a model with extra classes collapsed into EVAL_CLASSES:
    # "Museum+Collection": {
    #     "weights": os.path.join(BASE, "Models", "Museum+Collection_ResNet50_1", "best_weights_ResNet50.h5"),
    #     "classes": ["Carabidae", "Cerambycidae", "Chrysomelidae", "Corylophidae", "Curculionidae",
    #                 "Other", "Ptilodactylidae", "Scarabaeidae", "Staphylinidae"],
    #     "mapping": {"Cerambycidae": "Other", "Corylophidae": "Other", "Curculionidae": "Other",
    #                 "Ptilodactylidae": "Other", "Scarabaeidae": "Other"},
    # },


TARGET_CLASS = None          # a class name for its own ROC/PR plot, or None to skip and only get the cumulative plots
AVERAGING = "macro"          # "macro" (equal weight per class) or "micro" (equal weight per image)

img_height, img_width = (224, 224)
batch_size = 128
OUTPUT_DIR = os.path.join(BASE, "Presentation_Plots")
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")
# ================================================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


def sanitize(s):
    return re.sub(r'[^A-Za-z0-9]+', '', s)


# Every output filename below is tagged with the dataset + models that produced it, so
# re-running with a different DATASET_DIR/MODELS config creates new files instead of
# silently overwriting a previous run's plots.
DATASET_TAG = sanitize(os.path.basename(DATASET_DIR.rstrip(os.sep)))
MODELS_TAG = "_".join(sorted(sanitize(l) for l in MODELS.keys()))
RUN_TAG = f"{DATASET_TAG}__{MODELS_TAG}"


def cache_key(*parts):
    return hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def align_to_eval_classes(probs, model_classes, mapping, eval_classes):
    """Collapse a model's raw (n_samples, len(model_classes)) probabilities down to
    (n_samples, len(eval_classes)) by summing each model class's probability into
    whichever eval class it maps to (identity if not listed in `mapping`)."""
    aligned = np.zeros((probs.shape[0], len(eval_classes)))
    for j, mcls in enumerate(model_classes):
        target = mapping.get(mcls, mcls)
        if target not in eval_classes:
            raise ValueError(
                f"Model class '{mcls}' maps to '{target}', which isn't in EVAL_CLASSES {eval_classes}. "
                f"Add '{mcls}' to this model's 'mapping' in CONFIG."
            )
        aligned[:, eval_classes.index(target)] += probs[:, j]
    return aligned


def get_dataset_true_labels():
    """Scan DATASET_DIR once, return (generator, true_classes_in_eval_space)."""
    from tensorflow.keras.applications.resnet50 import preprocess_input
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    generator = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=False,
        class_mode='categorical')

    raw_class_names = list(generator.class_indices.keys())
    for name in raw_class_names:
        target = DATASET_LABEL_MAPPING.get(name, name)
        if target not in EVAL_CLASSES:
            raise ValueError(
                f"Dataset subfolder '{name}' maps to '{target}', which isn't in EVAL_CLASSES {EVAL_CLASSES}. "
                f"Add '{name}' to DATASET_LABEL_MAPPING in CONFIG."
            )
    raw_to_eval_idx = np.array([EVAL_CLASSES.index(DATASET_LABEL_MAPPING.get(n, n)) for n in raw_class_names])
    true_classes = raw_to_eval_idx[generator.classes]
    print(f"Dataset: {generator.samples} images, folders {raw_class_names} -> eval classes {EVAL_CLASSES}")
    return generator, true_classes


def get_model_predictions(label, cfg, generator):
    """Load cached aligned predictions for this model+dataset+eval-scheme combo, or
    compute (and cache) them fresh if nothing matches."""
    key = cache_key(label, cfg["weights"], DATASET_DIR, EVAL_CLASSES, sorted(cfg.get("mapping", {}).items()))
    cache_path = os.path.join(CACHE_DIR, f"{key}.npy")
    if os.path.exists(cache_path):
        print(f"[{label}] using cached predictions ({cache_path})")
        return np.load(cache_path)

    import tensorflow as tf
    print(f"[{label}] loading model from {cfg['weights']} ...")
    model = tf.keras.models.load_model(cfg["weights"])
    generator.reset()
    raw_preds = model.predict(generator, verbose=1)
    aligned = align_to_eval_classes(raw_preds, cfg["classes"], cfg.get("mapping", {}), EVAL_CLASSES)
    np.save(cache_path, aligned)
    return aligned


def macro_average_roc(true_classes, results):
    """One (fpr, tpr) curve per model, averaged equally across EVAL_CLASSES."""
    curves = {}
    for label, probs in results.items():
        all_fpr = np.unique(np.concatenate([
            roc_curve((true_classes == ci).astype(int), probs[:, ci])[0] for ci in range(len(EVAL_CLASSES))
        ]))
        mean_tpr = np.zeros_like(all_fpr)
        for ci in range(len(EVAL_CLASSES)):
            fpr, tpr, _ = roc_curve((true_classes == ci).astype(int), probs[:, ci])
            mean_tpr += np.interp(all_fpr, fpr, tpr)
        mean_tpr /= len(EVAL_CLASSES)
        curves[label] = (all_fpr, mean_tpr, auc(all_fpr, mean_tpr))
    return curves


def micro_average_roc(true_classes, results):
    """One (fpr, tpr) curve per model, pooling every (image, class) decision equally."""
    n_classes = len(EVAL_CLASSES)
    one_hot = np.eye(n_classes)[true_classes]
    curves = {}
    for label, probs in results.items():
        fpr, tpr, _ = roc_curve(one_hot.ravel(), probs.ravel())
        curves[label] = (fpr, tpr, auc(fpr, tpr))
    return curves


def macro_average_pr(true_classes, results):
    curves = {}
    for label, probs in results.items():
        all_recall = np.unique(np.concatenate([
            precision_recall_curve((true_classes == ci).astype(int), probs[:, ci])[1] for ci in range(len(EVAL_CLASSES))
        ]))
        mean_precision = np.zeros_like(all_recall)
        for ci in range(len(EVAL_CLASSES)):
            precision, recall, _ = precision_recall_curve((true_classes == ci).astype(int), probs[:, ci])
            # sklearn returns recall in descending order; np.interp needs ascending x
            mean_precision += np.interp(all_recall, recall[::-1], precision[::-1])
        mean_precision /= len(EVAL_CLASSES)
        curves[label] = (all_recall, mean_precision, auc(all_recall, mean_precision))
    return curves


def micro_average_pr(true_classes, results):
    n_classes = len(EVAL_CLASSES)
    one_hot = np.eye(n_classes)[true_classes]
    curves = {}
    for label, probs in results.items():
        precision, recall, _ = precision_recall_curve(one_hot.ravel(), probs.ravel())
        curves[label] = (recall, precision, auc(recall, precision))
    return curves


def plot_confusion_matrix(cm, labels, title, out_path):
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


def main():
    generator, true_classes = get_dataset_true_labels()
    results = {label: get_model_predictions(label, cfg, generator) for label, cfg in MODELS.items()}

    for label, probs in results.items():
        acc = np.mean(np.argmax(probs, axis=1) == true_classes)
        print(f"[{label}] accuracy: {acc:.4f}")

    # ---- Per-class ROC/PR for TARGET_CLASS (skipped if TARGET_CLASS is None) ----
    if TARGET_CLASS is not None:
        class_index = EVAL_CLASSES.index(TARGET_CLASS)
        class_true_labels = (true_classes == class_index).astype(int)

        plt.figure(figsize=(6, 5))
        for label, probs in results.items():
            fpr, tpr, _ = roc_curve(class_true_labels, probs[:, class_index])
            plt.plot(fpr, tpr, label=f'{label} (AUC={auc(fpr, tpr):.2f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
        plt.xlim([0, 1]); plt.ylim([0, 1.05])
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.title(f'ROC - {TARGET_CLASS}')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'ROC_{TARGET_CLASS}_{RUN_TAG}.png'), dpi=150)
        plt.close()

        plt.figure(figsize=(6, 5))
        for label, probs in results.items():
            precision, recall, _ = precision_recall_curve(class_true_labels, probs[:, class_index])
            plt.plot(recall, precision, label=f'{label} (AUC={auc(recall, precision):.2f})')
        plt.xlabel('Recall'); plt.ylabel('Precision')
        plt.title(f'Precision-Recall - {TARGET_CLASS}')
        plt.legend(loc='lower left')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'PR_{TARGET_CLASS}_{RUN_TAG}.png'), dpi=150)
        plt.close()

    # ---- Cumulative (micro/macro) ROC and PR: one curve per model ----
    roc_fn = micro_average_roc if AVERAGING == "micro" else macro_average_roc
    pr_fn = micro_average_pr if AVERAGING == "micro" else macro_average_pr

    plt.figure(figsize=(6, 5))
    for label, (x, y, area) in roc_fn(true_classes, results).items():
        plt.plot(x, y, label=f'{label} (AUC={area:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
    plt.xlim([0, 1]); plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(f'{AVERAGING.capitalize()}-averaged ROC (all {len(EVAL_CLASSES)} classes)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'ROC_{AVERAGING}_average_{RUN_TAG}.png'), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 5))
    for label, (x, y, area) in pr_fn(true_classes, results).items():
        plt.plot(x, y, label=f'{label} (AUC={area:.2f})')
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title(f'{AVERAGING.capitalize()}-averaged Precision-Recall (all {len(EVAL_CLASSES)} classes)')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'PR_{AVERAGING}_average_{RUN_TAG}.png'), dpi=150)
    plt.close()

    # ---- Confusion matrix per model ----
    for label, probs in results.items():
        cm = confusion_matrix(true_classes, np.argmax(probs, axis=1), labels=range(len(EVAL_CLASSES)))
        safe_label = sanitize(label)
        plot_confusion_matrix(cm, EVAL_CLASSES, f'Confusion Matrix - {label}',
                               os.path.join(OUTPUT_DIR, f'ConfusionMatrix_{safe_label}_{DATASET_TAG}.png'))

    print("\nDone. All figures saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
