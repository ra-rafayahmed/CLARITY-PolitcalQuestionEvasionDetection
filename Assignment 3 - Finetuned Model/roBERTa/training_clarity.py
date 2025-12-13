import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

from preprocess import QEvasionDataset  # your dataset class

# ------------------------------
# Configuration
# ------------------------------
MODEL_NAME = "roberta-base"
MAX_LEN = 128
BATCH_SIZE = 16
LR = 2e-5
EPOCHS = 3
WARMUP_RATIO = 0.06
NUM_LABELS = 3  # Clarity: Clear Reply, Clear Non-Reply, Ambivalent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "./saved_clarity_model"

# ------------------------------
# Load Tokenizer
# ------------------------------
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

# ------------------------------
# Load Datasets
# ------------------------------
def load_datasets():
    print("Loading QEvasion datasets for clarity...")
    train_dataset = QEvasionDataset(split="train", max_length=MAX_LEN)
    test_dataset = QEvasionDataset(split="test", max_length=MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    return train_loader, test_loader

# ------------------------------
# Model Setup
# ------------------------------
def load_model():
    print("Loading RoBERTa model for clarity classification...")
    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS
    )
    model.to(DEVICE)
    return model

# ------------------------------
# Training Loop
# ------------------------------
def train_model(model, train_loader, test_loader):
    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(WARMUP_RATIO * total_steps),
        num_training_steps=total_steps
    )
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch in loop:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["clarity_label"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            loop.set_postfix({"loss": loss.item()})

        avg_loss = epoch_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} average loss: {avg_loss:.4f}")

        evaluate(model, test_loader)

    save_model(model)

# ------------------------------
# Evaluation Function
# ------------------------------
def evaluate(model, data_loader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["clarity_label"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    print(f"\n--- Validation Results ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, zero_division=0))
    model.train()

# ------------------------------
# Save Model
# ------------------------------
def save_model(model):
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"Clarity model saved to {SAVE_DIR}")

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    train_loader, test_loader = load_datasets()
    model = load_model()
    train_model(model, train_loader, test_loader)
