# baseline.py
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# ------------------------------
# Configuration
# ------------------------------
MODEL_NAME = "roberta-base"
NUM_LABELS = 9
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------
# Load Tokenizer and Model
# ------------------------------
def get_tokenizer():
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    return tokenizer

def get_model():
    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS
    )
    model.to(DEVICE)
    return model
