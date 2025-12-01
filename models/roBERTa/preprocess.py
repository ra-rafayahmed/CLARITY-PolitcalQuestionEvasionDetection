import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import RobertaTokenizer

# ------------------------------
# Label mappings
# ------------------------------
CLARITY_MAP = {
    "Clear Reply": 0,
    "Clear Non-Reply": 1,
    "Ambivalent Reply": 2,
    "Ambivalent": 2           # fix for HF mismatch
}

EVASION_MAP = {
    "Explicit": 0,
    "Implicit": 1,
    "Dodging": 2,
    "General": 3,
    "Deflection": 4,
    "Partial/half-answer": 5,
    "Declining to answer": 6,
    "Claims ignorance": 7,
    "Clarification": 8
}


class QEvasionDataset(Dataset):
    def __init__(self, split="train", tokenizer_name="roberta-base", max_length=128):
        print(f"Loading QEvasion HuggingFace dataset...")

        # Load full dataset (only one split exists)
        raw = load_dataset("ailsntua/QEvasion")["train"]

        # Filter invalid rows
        filtered = []
        for row in raw:
            if row["clarity_label"] in CLARITY_MAP and row["evasion_label"] in EVASION_MAP:
                filtered.append(row)

        self.data = filtered
        print(f"Total valid samples: {len(self.data)}")

        # ---- MANUAL SPLITTING ----
        split_point = int(len(self.data) * 0.90)
        train_data = self.data[:split_point]
        test_data = self.data[split_point:]

        if split == "train":
            self.data = train_data
        elif split == "test":
            self.data = test_data
        else:
            raise ValueError(f"Unknown split {split}")

        print(f"{split.upper()} samples: {len(self.data)}")

        self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]

        text = (row["interview_question"] or "") + " " + (row["interview_answer"] or "")

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        clarity_label = CLARITY_MAP[row["clarity_label"]]
        evasion_label = EVASION_MAP[row["evasion_label"]]

        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["clarity_label"] = torch.tensor(clarity_label, dtype=torch.long)
        item["evasion_label"] = torch.tensor(evasion_label, dtype=torch.long)

        return item
