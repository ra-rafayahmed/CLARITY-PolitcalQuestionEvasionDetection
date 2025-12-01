# training.py
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import RobertaForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import os

from preprocess import QEvasionDataset  # Updated dataset class

# ------------------------------
# Configuration
# ------------------------------
MODEL_NAME = "roberta-base"
MAX_LEN = 128
BATCH_SIZE = 16
LR = 2e-5
EPOCHS = 3
WARMUP_RATIO = 0.06
NUM_LABELS = 9
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------
# Load datasets
# ------------------------------
def load_datasets():
    print("Loading QEvasion datasets...")
    train_dataset = QEvasionDataset(split="train", max_length=MAX_LEN)
    test_dataset  = QEvasionDataset(split="test", max_length=MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    return train_loader, test_loader

# ------------------------------
# Load model
# ------------------------------
def load_model():
    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS
    )
    model.to(DEVICE)
    model.train()
    return model

# ------------------------------
# Evaluation (both clarity & evasion)
# ------------------------------
def evaluate(model, data_loader):
    model.eval()
    all_preds_evasion = []
    all_labels_evasion = []
    all_labels_clarity = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels_evasion = batch["evasion_label"].to(DEVICE)
            labels_clarity = batch["clarity_label"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            all_preds_evasion.extend(preds.cpu().numpy())
            all_labels_evasion.extend(labels_evasion.cpu().numpy())
            all_labels_clarity.extend(labels_clarity.cpu().numpy())

    # Evasion metrics
    evasion_acc = accuracy_score(all_labels_evasion, all_preds_evasion)
    evasion_macro_f1 = f1_score(all_labels_evasion, all_preds_evasion, average="macro")

    # Clarity metrics (for reference)
    clarity_acc = accuracy_score(all_labels_clarity, all_labels_clarity)  # just placeholder
    clarity_macro_f1 = f1_score(all_labels_clarity, all_labels_clarity, average="macro")  # placeholder

    print(f"Eval Evasion Accuracy:  {evasion_acc:.4f}")
    print(f"Eval Evasion Macro F1: {evasion_macro_f1:.4f}")
    print(f"Eval Clarity (ref) Accuracy:  {clarity_acc:.4f}")
    print(f"Eval Clarity (ref) Macro F1: {clarity_macro_f1:.4f}\n")
    model.train()

# ------------------------------
# Training loop
# ------------------------------
def train_model(model, train_loader, test_loader):
    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(WARMUP_RATIO * total_steps),
        num_training_steps=total_steps
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    print("\nStarting training...\n")
    for epoch in range(EPOCHS):
        epoch_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch in loop:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["evasion_label"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            loop.set_postfix({"loss": loss.item()})

        avg_loss = epoch_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} average loss: {avg_loss:.4f}")

        print("\nRunning evaluation...")
        evaluate(model, test_loader)

    print("\nTraining complete.")
    save_model(model)

# ------------------------------
# Save model
# ------------------------------
def save_model(model):
    save_dir = "./saved_model"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    print(f"Model saved to {save_dir}")

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    train_loader, test_loader = load_datasets()
    model = load_model()
    train_model(model, train_loader, test_loader)
