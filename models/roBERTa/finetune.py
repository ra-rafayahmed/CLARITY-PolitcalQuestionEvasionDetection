# finetune_roberta.py
import os
import random
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import RobertaTokenizer, RobertaModel, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import nltk
from nltk.corpus import wordnet

# ------------------------------
# Ensure NLTK wordnet is downloaded
# ------------------------------
nltk.download('wordnet', quiet=True)

def synonym_replacement(text, n=1):
    words = text.split()
    for _ in range(n):
        idx = random.randint(0, len(words)-1)
        syns = wordnet.synsets(words[idx])
        if syns:
            lemmas = [l.name() for s in syns for l in s.lemmas() if l.name() != words[idx]]
            if lemmas:
                words[idx] = random.choice(lemmas)
    return " ".join(words)

# ------------------------------
# Dataset
# ------------------------------
from datasets import load_dataset

CLARITY_MAP = {"Clear Reply": 0, "Clear Non-Reply": 1, "Ambivalent Reply": 2}
EVASION_MAP = {
    "Explicit": 0, "Implicit": 1, "Dodging": 2, "General": 3, "Deflection": 4,
    "Partial/half-answer": 5, "Declining to answer": 6, "Claims ignorance": 7, "Clarification": 8
}

class QEvasionDataset(torch.utils.data.Dataset):
    def __init__(self, split="train", tokenizer_name="roberta-base", max_length=128, augment=False):
        self.augment = augment
        dataset = load_dataset("ailsntua/QEvasion")["train"]
        filtered = [row for row in dataset if row["clarity_label"] in CLARITY_MAP and row["evasion_label"] in EVASION_MAP]
        split_point = int(0.9 * len(filtered))
        self.data = filtered[:split_point] if split=="train" else filtered[split_point:]
        self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        print(f"{split.upper()} samples: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        text = (row["interview_question"] or "") + " " + (row["interview_answer"] or "")
        if self.augment and random.random() < 0.3:
            text = synonym_replacement(text, n=1)
        encoding = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
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
            pooled = outputs.last_hidden_state[:,0,:]  # CLS token fallback
        pooled = self.dropout(pooled)
        clarity_logits = self.clarity_head(pooled)
        evasion_logits = self.evasion_head(pooled)

        loss = None
        if clarity_labels is not None and evasion_labels is not None:
            # Weighted cross-entropy
            clarity_weights = torch.tensor([1.0,1.0,1.0]).to(clarity_labels.device)
            evasion_weights = torch.ones(len(EVASION_MAP)).to(evasion_labels.device)
            loss_fn = nn.CrossEntropyLoss(weight=clarity_weights)
            loss_clarity = loss_fn(clarity_logits, clarity_labels)
            loss_fn_e = nn.CrossEntropyLoss(weight=evasion_weights)
            loss_evasion = loss_fn_e(evasion_logits, evasion_labels)
            loss = loss_clarity + loss_evasion
        return clarity_logits, evasion_logits, loss

# ------------------------------
# Training setup
# ------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS = 5
LR = 1e-5
MAX_LEN = 128
SAVE_DIR = "./saved_multitask_model"
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Using device: {DEVICE}")

train_dataset = QEvasionDataset(split="train", max_length=MAX_LEN, augment=True)
test_dataset  = QEvasionDataset(split="test", max_length=MAX_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE)

model = MultiTaskRoberta().to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LR)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)

# ------------------------------
# Evaluation function
# ------------------------------
def evaluate(model, loader):
    model.eval()
    all_clarity, all_evasion = [], []
    pred_clarity, pred_evasion = [], []

    with torch.no_grad():
        for batch in loader:
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

    clarity_acc = accuracy_score(all_clarity, pred_clarity)
    clarity_macro = f1_score(all_clarity, pred_clarity, average="macro")
    clarity_weighted = f1_score(all_clarity, pred_clarity, average="weighted")
    evasion_acc = accuracy_score(all_evasion, pred_evasion)
    evasion_macro = f1_score(all_evasion, pred_evasion, average="macro")
    evasion_weighted = f1_score(all_evasion, pred_evasion, average="weighted")

    print(f"Clarity - Accuracy: {clarity_acc:.4f}, Macro F1: {clarity_macro:.4f}, Weighted F1: {clarity_weighted:.4f}")
    print(f"Evasion - Accuracy: {evasion_acc:.4f}, Macro F1: {evasion_macro:.4f}, Weighted F1: {evasion_weighted:.4f}")

    # Convert to native Python int for JSON
    predictions = {
        "clarity": [int(x) for x in pred_clarity],
        "evasion": [int(x) for x in pred_evasion],
        "clarity_labels": [int(x) for x in all_clarity],
        "evasion_labels": [int(x) for x in all_evasion]
    }

    return (clarity_acc, clarity_macro, clarity_weighted), (evasion_acc, evasion_macro, evasion_weighted), predictions

# ------------------------------
# Training loop
# ------------------------------
all_epoch_metrics = []

for epoch in range(EPOCHS):
    model.train()
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    total_loss = 0
    for batch in loop:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        clarity_labels = batch["clarity_label"].to(DEVICE)
        evasion_labels = batch["evasion_label"].to(DEVICE)

        _, _, loss = model(input_ids, attention_mask, clarity_labels, evasion_labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        loop.set_postfix({"loss": loss.item()})
    avg_loss = total_loss / len(train_loader)
    print(f"\nEpoch {epoch+1} Avg Loss: {avg_loss:.4f}")

    print("\n--- Evaluation ---")
    clarity_metrics, evasion_metrics, predictions = evaluate(model, test_loader)
    epoch_info = {
        "epoch": epoch+1,
        "avg_loss": float(avg_loss),
        "clarity": {
            "accuracy": float(clarity_metrics[0]),
            "macro_f1": float(clarity_metrics[1]),
            "weighted_f1": float(clarity_metrics[2])
        },
        "evasion": {
            "accuracy": float(evasion_metrics[0]),
            "macro_f1": float(evasion_metrics[1]),
            "weighted_f1": float(evasion_metrics[2])
        },
        "predictions": predictions
    }
    all_epoch_metrics.append(epoch_info)

# ------------------------------
# Save model & metrics
# ------------------------------
model_path = os.path.join(SAVE_DIR, "multitask_roberta")
model.roberta.save_pretrained(model_path)
torch.save(model.state_dict(), os.path.join(model_path, "multitask_head.pt"))

# Save metrics & predictions to JSON
with open(os.path.join(model_path, "metrics_predictions.json"), "w") as f:
    json.dump(all_epoch_metrics, f, indent=4)

print(f"Model, tokenizer, and metrics saved to {model_path}")
