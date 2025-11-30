import os
import math
import json
import random
from tqdm import tqdm
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

from ..data_loader import load_data
from ..preprocess.tokenizer import build_vocab, texts_to_sequences, pad_sequences, save_vocab
from ..model.bilstm_model import BiLSTMClassifier


def download_and_load_glove(dim=300, glove_dir="glove"):
    """Downloads GloVe 6B embeddings if not present and loads them into a dict.
    This function attempts to download from the Stanford NLP page. If running in an offline environment,
    manually download and place the file at the expected path: glove/glove.6B.300d.txt
    """
    import zipfile
    import urllib.request

    os.makedirs(glove_dir, exist_ok=True)
    glove_txt = os.path.join(glove_dir, f"glove.6B.{dim}d.txt")
    if not os.path.exists(glove_txt):
        print("GloVe not found locally. Downloading (approx 0.9GB zip) — this may take a while...")
        url = "http://nlp.stanford.edu/data/glove.6B.zip"
        zip_path = os.path.join(glove_dir, "glove6b.zip")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(glove_dir)
        os.remove(zip_path)
    # load into dict
    emb_index = {}
    with open(glove_txt, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            emb_index[word] = vector
    print(f"Loaded {len(emb_index)} word vectors from GloVe")
    return emb_index


def build_embedding_matrix(vocab, emb_index, embed_dim=300):
    vocab_size = len(vocab)
    embedding_matrix = np.zeros((vocab_size, embed_dim), dtype=np.float32)
    unk_count = 0
    for word, idx in vocab.items():
        if word in emb_index:
            embedding_matrix[idx] = emb_index[word]
        else:
            # random init for OOV
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embed_dim,))
            unk_count += 1
    print(f"Embedding matrix built. OOV words: {unk_count}/{vocab_size}")
    return torch.tensor(embedding_matrix)


def prepare_examples(dataset, vocab, max_len=512):
    # dataset is HF dataset containing interview_question and interview_answer and clarity_label
    texts = []
    labels = []
    for ex in dataset:
        q = ex.get("interview_question") or ""
        a = ex.get("interview_answer") or ""
        text = q.strip() + " [SEP] " + a.strip()
        texts.append(text)
        # clarity_label may be e.g., 'Clear Reply', 'Ambivalent Reply', 'Clear Non-Reply'
        labels.append(ex.get("clarity_label"))

    # map label strings to ints
    unique_labels = sorted(list(set(labels)))
    label2id = {lbl: i for i, lbl in enumerate(unique_labels)}
    y = [label2id[l] for l in labels]

    seqs = texts_to_sequences(texts, vocab)
    seqs = pad_sequences(seqs, max_len=max_len)

    X = np.array(seqs, dtype=np.int64)
    y = np.array(y, dtype=np.int64)
    # attention mask
    att = (X != 0).astype(np.int64)
    return X, att, y, label2id


def train():
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    root = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(root)
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("Loading dataset...")
    train_ds, test_ds = load_data()

    print("Preparing texts to build vocab (train split only)...")
    texts_for_vocab = []
    for ex in train_ds:
        q = ex.get("interview_question") or ""
        a = ex.get("interview_answer") or ""
        texts_for_vocab.append((q.strip() + " [SEP] " + a.strip()))

    print("Building vocabulary...")
    vocab = build_vocab(texts_for_vocab, vocab_size=30000)
    save_vocab(vocab, os.path.join(project_root, "vocab.json"))

    print("Loading GloVe... (this may take a while if not present)")
    emb_index = download_and_load_glove(dim=300, glove_dir=os.path.join(project_root, "glove"))
    embedding_weights = build_embedding_matrix(vocab, emb_index, embed_dim=300)

    print("Preparing train / val / test arrays...")
    X_train, att_train, y_train, label2id = prepare_examples(train_ds, vocab, max_len=512)
    X_test, att_test, y_test, _ = prepare_examples(test_ds, vocab, max_len=512)

    # train/val split from train
    X_tr, X_val, A_tr, A_val, y_tr, y_val = train_test_split(X_train, att_train, y_train, test_size=0.15, random_state=42, stratify=y_train)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    batch_size = 16
    train_dataset = TensorDataset(torch.tensor(X_tr), torch.tensor(A_tr), torch.tensor(y_tr))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(A_val), torch.tensor(y_val))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(att_test), torch.tensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = BiLSTMClassifier(vocab_size=len(vocab), embed_dim=300, hidden_size=128, num_classes=len(label2id), embedding_weights=embedding_weights, dropout=0.3)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    num_epochs = 8

    best_val_f1 = 0.0
    history = {"train_loss": [], "val_loss": [], "val_f1": []}

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            xb, attb, yb = [b.to(device) for b in batch]
            optimizer.zero_grad()
            logits = model(xb, attention_mask=attb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        avg_train_loss = total_loss / len(train_loader.dataset)

        # validation
        model.eval()
        y_val_true = []
        y_val_pred = []
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                xb, attb, yb = [b.to(device) for b in batch]
                logits = model(xb, attention_mask=attb)
                loss = F.cross_entropy(logits, yb)
                val_loss += loss.item() * xb.size(0)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                y_val_pred.extend(preds.tolist())
                y_val_true.extend(yb.cpu().numpy().tolist())

        avg_val_loss = val_loss / len(val_loader.dataset)
        precision, recall, f1, _ = precision_recall_fscore_support(y_val_true, y_val_pred, average="macro", zero_division=0)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_f1"].append(f1)

        print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} val_f1={f1:.4f}")

        # save best
        if f1 > best_val_f1:
            best_val_f1 = f1
            torch.save(model.state_dict(), os.path.join(results_dir, "best_model.pt"))
            with open(os.path.join(results_dir, "label2id.json"), "w") as f:
                json.dump(label2id, f)

    # final evaluation on test
    print("Evaluating on test set...")
    model.load_state_dict(torch.load(os.path.join(results_dir, "best_model.pt")))
    model.to(device)
    y_test_true = []
    y_test_pred = []
    with torch.no_grad():
        for batch in test_loader:
            xb, attb, yb = [b.to(device) for b in batch]
            logits = model(xb, attention_mask=attb)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_test_pred.extend(preds.tolist())
            y_test_true.extend(yb.cpu().numpy().tolist())

    acc = accuracy_score(y_test_true, y_test_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_test_true, y_test_pred, average="macro", zero_division=0)
    report = classification_report(y_test_true, y_test_pred, zero_division=0)
    cm = confusion_matrix(y_test_true, y_test_pred)

    metrics = {
        "accuracy": acc,
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1)
    }

    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(results_dir, "classification_report.txt"), "w") as f:
        f.write(report)
    np.savetxt(os.path.join(results_dir, "confusion_matrix.csv"), cm, fmt="%d", delimiter=",")
    print("Saved metrics to", results_dir)


if __name__ == "__main__":
    train()

