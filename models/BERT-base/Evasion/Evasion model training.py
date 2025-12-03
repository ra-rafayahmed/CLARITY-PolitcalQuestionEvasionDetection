from datasets import load_dataset
import pandas as pd

# Load full dataset
dataset = load_dataset("ailsntua/QEvasion")

# Convert train dataset to pandas
df_full = pd.DataFrame(dataset['train'])
print(df_full.head())

from sklearn.model_selection import train_test_split

# Split 80% train, 20% test
train_df, test_df = train_test_split(df_full, test_size=0.2, random_state=42)

# Reset indices
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print(len(train_df), len(test_df))

unique_labels = ['Claims ignorance', 'Clarification', 'Declining to answer', 
                 'Deflection', 'Dodging', 'Explicit', 'General', 'Implicit', 'Partial/half-answer']

label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}
num_labels = len(unique_labels)

# Map textual labels to numbers
train_df['label'] = train_df['evasion_label'].map(label2id)
test_df['label'] = test_df['evasion_label'].map(label2id)

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch['interview_answer'], padding='max_length', truncation=True, max_length=256)

# Apply tokenizer
train_encodings = tokenizer(list(train_df['interview_answer']), truncation=True, padding=True, max_length=256)
test_encodings = tokenizer(list(test_df['interview_answer']), truncation=True, padding=True, max_length=256)

import torch

class EvasionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = EvasionDataset(train_encodings, list(train_df['label']))
test_dataset = EvasionDataset(test_encodings, list(test_df['label']))

from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=num_labels)
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results_evasion",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs_evasion",
    logging_steps=10,
    save_total_limit=2,
    save_steps=500,
    report_to="none"  # disables wandb login prompt
)

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_weighted": f1_score(labels, preds, average='weighted'),
        "precision_weighted": precision_score(labels, preds, average='weighted'),
        "recall_weighted": recall_score(labels, preds, average='weighted')
    }

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

train_output = trainer.train()
print(train_output)

metrics = trainer.evaluate(eval_dataset=test_dataset)
print(metrics)

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import json
from google.colab import files

# -------------------------
# 1️⃣ Overall metrics bar plot
# -------------------------
plt.figure(figsize=(6,4))
plt.bar(
    ['Eval Loss','Accuracy','F1'],
    [metrics['eval_loss'], metrics['eval_accuracy'], metrics['eval_f1_weighted']],
    color=['red','green','blue']
)
plt.ylim(0,1)
plt.title("Evasion Task Metrics")
plt.savefig("evasion_metrics_plot.png")
plt.show()
files.download("evasion_metrics_plot.png")  # download to local

# -------------------------
# 2️⃣ Confusion matrix
# -------------------------
preds = trainer.predict(test_dataset).predictions.argmax(-1)
labels = trainer.predict(test_dataset).label_ids

cm = confusion_matrix(labels, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)

plt.figure(figsize=(10,10))
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Evasion Task Confusion Matrix")
plt.savefig("evasion_confusion_matrix.png")
plt.show()
files.download("evasion_confusion_matrix.png")  # download to local

# -------------------------
# 3️⃣ Per-class precision, recall, F1
# -------------------------
report = classification_report(labels, preds, target_names=unique_labels, output_dict=True)

labels_names = unique_labels
precision = [report[label]['precision'] for label in labels_names]
recall = [report[label]['recall'] for label in labels_names]
f1 = [report[label]['f1-score'] for label in labels_names]

x = range(len(labels_names))
plt.figure(figsize=(10,6))
plt.bar(x, precision, width=0.2, label='Precision', align='center')
plt.bar([i+0.2 for i in x], recall, width=0.2, label='Recall', align='center')
plt.bar([i+0.4 for i in x], f1, width=0.2, label='F1-score', align='center')
plt.xticks([i+0.2 for i in x], labels_names, rotation=45)
plt.ylim(0,1)
plt.title("Evasion Task Metrics per Class")
plt.ylabel("Score")
plt.legend()
plt.tight_layout()
plt.savefig("evasion_per_class_metrics.png")
plt.show()
files.download("evasion_per_class_metrics.png")  # download to local

# -------------------------
# 4️⃣ Save metrics and classification report as JSON
# -------------------------
with open("evasion_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
files.download("evasion_metrics.json")  # download to local

with open("evasion_class_report.json", "w") as f:
    json.dump(report, f, indent=4)
files.download("evasion_class_report.json")  # download to local






























































