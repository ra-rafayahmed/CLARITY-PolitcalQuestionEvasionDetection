# realtime_inference.py
import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel
import os

# ------------------------------
# Define your multi-task model
# ------------------------------
CLARITY_MAP = {0: "Clear Reply", 1: "Clear Non-Reply", 2: "Ambivalent Reply"}
EVASION_MAP = {
    0: "Explicit", 1: "Implicit", 2: "Dodging", 3: "General", 4: "Deflection",
    5: "Partial/half-answer", 6: "Declining to answer", 7: "Claims ignorance", 8: "Clarification"
}

class MultiTaskRoberta(nn.Module):
    def __init__(self, model_name="roberta-large", hidden_dropout=0.2):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        hidden_size = self.roberta.config.hidden_size
        self.dropout = nn.Dropout(hidden_dropout)
        self.clarity_head = nn.Linear(hidden_size, len(CLARITY_MAP))
        self.evasion_head = nn.Linear(hidden_size, len(EVASION_MAP))

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        if pooled is None:
            pooled = outputs.last_hidden_state[:,0,:]  # fallback to CLS token
        pooled = self.dropout(pooled)
        clarity_logits = self.clarity_head(pooled)
        evasion_logits = self.evasion_head(pooled)
        return clarity_logits, evasion_logits

# ------------------------------
# Load model & tokenizer
# ------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "./saved_multitask_model/multitask_roberta"

# Fixed: use original tokenizer since saved folder doesn't have it
tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

# Load model
model = MultiTaskRoberta(model_name="roberta-large").to(DEVICE)
state_dict_path = os.path.join(MODEL_PATH, "multitask_head.pt")
model.load_state_dict(torch.load(state_dict_path, map_location=DEVICE))
model.eval()

# ------------------------------
# Function to predict clarity & evasion
# ------------------------------
def predict(text):
    encoding = tokenizer(text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    with torch.no_grad():
        c_logits, e_logits = model(input_ids, attention_mask)
        c_pred = torch.argmax(c_logits, dim=1).item()
        e_pred = torch.argmax(e_logits, dim=1).item()

    clarity_label = CLARITY_MAP[c_pred]
    evasion_label = EVASION_MAP[e_pred]
    return clarity_label, evasion_label

# ------------------------------
# User input loop
# ------------------------------
if __name__ == "__main__":
    print("=== Real-time Clarity & Evasion Prediction ===")
    while True:
        user_input = input("\nEnter interview question + answer (or type 'exit' to quit):\n> ")
        if user_input.lower() in ["exit", "quit"]:
            break

        clarity, evasion = predict(user_input)
        print(f"\nPredicted Clarity: {clarity}")
        print(f"Predicted Evasion: {evasion}")
