# evaluation.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaModel
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ------------------------------
# Dataset & Mapping
# ------------------------------
from datasets import load_dataset

CLARITY_MAP = {"Clear Reply": 0, "Clear Non-Reply": 1, "Ambivalent Reply": 2}
EVASION_MAP = {
    "Explicit": 0, "Implicit": 1, "Dodging": 2, "General": 3, "Deflection": 4,
    "Partial/half-answer": 5, "Declining to answer": 6, "Claims ignorance": 7, "Clarification": 8
}

class QEvasionDataset(torch.utils.data.Dataset):
    def __init__(self, split="test", tokenizer_name="roberta-base", max_length=128):
        dataset = load_dataset("ailsntua/QEvasion")["train"]
        filtered = [row for row in dataset if row["clarity_label"] in CLARITY_MAP and row["evasion_label"] in EVASION_MAP]
        split_point = int(0.9 * len(filtered))
        self.data = filtered[split_point:] if split=="test" else filtered[:split_point]
        self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        print(f"{split.upper()} samples: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        text = (row["interview_question"] or "") + " " + (row["interview_answer"] or "")
        encoding = self.tokenizer(text, truncation=True, padding="max_length",
                                  max_length=self.max_length, return_tensors="pt")
        item = {k: v.squeeze(0) for k,v in encoding.items()}
        item["clarity_label"] = torch.tensor(CLARITY_MAP[row["clarity_label"]], dtype=torch.long)
        item["evasion_label"] = torch.tensor(EVASION_MAP[row["evasion_label"]], dtype=torch.long)
        return item

# ------------------------------
# Multi-task Model
# ------------------------------
class MultiTaskRoberta(nn.Module):
    def __init__(self, model_name="roberta-base", hidden_dropout=0.2):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        hidden_size = self.roberta.config.hidden_size
        self.dropout = nn.Dropout(hidden_dropout)
        self.clarity_head = nn.Linear(hidden_size, len(CLARITY_MAP))
        self.evasion_head = nn.Linear(hidden_size, len(EVASION_MAP))

    def forward(self, input_ids, attention_mask, clarity_labels=None, evasion_labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        if pooled is None:
            pooled = outputs.last_hidden_state[:,0,:]  # CLS fallback
        pooled = self.dropout(pooled)
        clarity_logits = self.clarity_head(pooled)
        evasion_logits = self.evasion_head(pooled)
        loss = None
        if clarity_labels is not None and evasion_labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(clarity_logits, clarity_labels) + loss_fn(evasion_logits, evasion_labels)
        return clarity_logits, evasion_logits, loss

# ------------------------------
# Evaluation
# ------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
MAX_LEN = 128
MODEL_PATH = "./saved_multitask_model/multitask_roberta"

# Load dataset
test_dataset = QEvasionDataset(split="test", max_length=MAX_LEN)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Load model
model = MultiTaskRoberta().to(DEVICE)
state_dict = torch.load(os.path.join(MODEL_PATH, "multitask_head.pt"), map_location=DEVICE)
model.load_state_dict(state_dict, strict=False)
model.eval()

all_clarity, all_evasion = [], []
pred_clarity, pred_evasion = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        c_labels = batch["clarity_label"].to(DEVICE)
        e_labels = batch["evasion_label"].to(DEVICE)

        c_logits, e_logits, _ = model(input_ids, attention_mask)
        c_preds = torch.argmax(c_logits, dim=1)
        e_preds = torch.argmax(e_logits, dim=1)

        all_clarity.extend(c_labels.cpu().numpy())
        pred_clarity.extend(c_preds.cpu().numpy())
        all_evasion.extend(e_labels.cpu().numpy())
        pred_evasion.extend(e_preds.cpu().numpy())

# Metrics
clarity_acc = accuracy_score(all_clarity, pred_clarity)
clarity_macro = f1_score(all_clarity, pred_clarity, average="macro")
clarity_weighted = f1_score(all_clarity, pred_clarity, average="weighted")
evasion_acc = accuracy_score(all_evasion, pred_evasion)
evasion_macro = f1_score(all_evasion, pred_evasion, average="macro")
evasion_weighted = f1_score(all_evasion, pred_evasion, average="weighted")

print("=== Overall Metrics ===")
print(f"Clarity - Accuracy: {clarity_acc:.4f}, Macro F1: {clarity_macro:.4f}, Weighted F1: {clarity_weighted:.4f}")
print(f"Evasion - Accuracy: {evasion_acc:.4f}, Macro F1: {evasion_macro:.4f}, Weighted F1: {evasion_weighted:.4f}")

# Classification reports (handle missing classes)
clarity_labels_list = list(CLARITY_MAP.values())
evasion_labels_list = list(EVASION_MAP.values())

print("\n=== Per-class Classification Report ===")
print("Clarity:\n", classification_report(all_clarity, pred_clarity, labels=clarity_labels_list, target_names=CLARITY_MAP.keys(), zero_division=0))
print("Evasion:\n", classification_report(all_evasion, pred_evasion, labels=evasion_labels_list, target_names=EVASION_MAP.keys(), zero_division=0))

# ------------------------------
# Confusion matrices
# ------------------------------
def plot_confusion(y_true, y_pred, classes, title, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()  # close the figure to avoid overlapping plots

# Save confusion matrices
plot_confusion(all_clarity, pred_clarity, list(CLARITY_MAP.keys()), 
               "Clarity Confusion Matrix", os.path.join(MODEL_PATH, "clarity_confusion.png"))
plot_confusion(all_evasion, pred_evasion, list(EVASION_MAP.keys()), 
               "Evasion Confusion Matrix", os.path.join(MODEL_PATH, "evasion_confusion.png"))

# Save metrics
metrics_path = os.path.join(MODEL_PATH, "evaluation_metrics.json")
metrics = {
    "clarity": {"accuracy": clarity_acc, "macro_f1": clarity_macro, "weighted_f1": clarity_weighted},
    "evasion": {"accuracy": evasion_acc, "macro_f1": evasion_macro, "weighted_f1": evasion_weighted}
}
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)

print(f"\nMetrics saved to {metrics_path}")
print(f"Confusion matrices saved to {MODEL_PATH}")
