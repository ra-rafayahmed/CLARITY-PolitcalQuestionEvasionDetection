import os
import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from preprocess import QEvasionDataset
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import json

# ------------------------------
# Configuration
# ------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
CLARITY_MODEL_DIR = "./saved_clarity_model"
EVASION_MODEL_DIR = "./saved_model"
PLOTS_DIR = "./plots"

# Ensure plots directory exists
os.makedirs(PLOTS_DIR, exist_ok=True)

# ------------------------------
# Load Tokenizers & Models
# ------------------------------
tokenizer_clarity = RobertaTokenizer.from_pretrained(CLARITY_MODEL_DIR, local_files_only=True)
tokenizer_evasion = RobertaTokenizer.from_pretrained(EVASION_MODEL_DIR, local_files_only=True)

clarity_model = RobertaForSequenceClassification.from_pretrained(CLARITY_MODEL_DIR)
clarity_model.to(DEVICE)
clarity_model.eval()

evasion_model = RobertaForSequenceClassification.from_pretrained(EVASION_MODEL_DIR)
evasion_model.to(DEVICE)
evasion_model.eval()

# ------------------------------
# Load Test Dataset
# ------------------------------
test_dataset = QEvasionDataset(split="test", max_length=128)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

print(f"Test samples: {len(test_dataset)}")
if len(test_dataset) == 0:
    raise ValueError("Test dataset is empty! Check preprocessing.")

# ------------------------------
# Evaluation Function
# ------------------------------
def evaluate(model, label_name, model_name):
    all_labels = []
    all_preds = []
    all_confidences = []

    for batch in test_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch[label_name].to(DEVICE)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_confidences.extend(probs.max(dim=1).values.cpu().numpy())

    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    avg_conf = np.mean(all_confidences)

    print(f"\n=== {model_name.upper()} EVALUATION RESULTS ===")
    print(f"Accuracy:      {acc:.4f}")
    print(f"Macro F1:      {macro_f1:.4f}")
    print(f"Avg Confidence:{avg_conf:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, zero_division=0))

    # Save confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10,8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"{model_name} Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(PLOTS_DIR, f"confusion_matrix_{model_name}.png"))
    plt.close()

    # Save metrics to JSON
    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "avg_confidence": float(avg_conf),
        "classification_report": classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    }
    with open(os.path.join(PLOTS_DIR, f"evaluation_results_{model_name}.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Print sample predictions
    print("\nSample Predictions:")
    for i in range(min(10, len(all_labels))):
        print(f"Sample {i+1}: True Label: {all_labels[i]}, Predicted Label: {all_preds[i]}, Confidence: {all_confidences[i]:.4f}")


# ------------------------------
# Run Evaluation
# ------------------------------
evaluate(clarity_model, "clarity_label", "clarity")
evaluate(evasion_model, "evasion_label", "evasion")
