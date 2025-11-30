import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from ..model.bilstm_model import BiLSTMClassifier


def plot_confusion(cm, labels, out_path):
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def load_label_map(results_dir):
    with open(os.path.join(results_dir, "label2id.json"), "r") as f:
        label2id = json.load(f)
    id2label = {int(v): k for k, v in label2id.items()}
    labels = [id2label[i] for i in sorted(id2label.keys())]
    return labels


def evaluate():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(root, "results")
    # load predictions saved by training script (confusion matrix etc.)
    cm = np.loadtxt(os.path.join(results_dir, "confusion_matrix.csv"), delimiter=",")
    labels = load_label_map(results_dir)
    out_png = os.path.join(results_dir, "confusion_matrix.png")
    plot_confusion(cm, labels, out_png)
    print("Saved confusion matrix to", out_png)
    with open(os.path.join(results_dir, "metrics.json"), "r") as f:
        metrics = json.load(f)
    print(metrics)


if __name__ == "__main__":
    evaluate()

