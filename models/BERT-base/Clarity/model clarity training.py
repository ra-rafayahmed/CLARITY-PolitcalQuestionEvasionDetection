from datasets import load_dataset

dataset = load_dataset("ailsntua/QEvasion")
print(dataset)
print(dataset['train'][0])
LARITY_MAP = {
    "Clear Reply": 0,
    "Clear Non-Reply": 1,
    "Ambivalent": 2
}
def map_clarity_labels(example):
    example["labels"] = CLARITY_MAP[example["clarity_label"]]
    return example

dataset = dataset.map(map_clarity_labels)
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(example):
    return tokenizer(example["interview_answer"], 
                     padding="max_length", 
                     truncation=True,
                     max_length=256)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
import torch

tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

train_dataset = tokenized_datasets["train"]
test_dataset = tokenized_datasets["test"]

from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=3)  # 3 classes for clarity
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    report_to="none"   # disables W&B and other logging integrations
)
from sklearn.metrics import f1_score, accuracy_score

def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

metrics = trainer.evaluate(eval_dataset=test_dataset)
print(metrics)

from sklearn.metrics import classification_report

# Get predictions from trainer
preds = trainer.predict(test_dataset).predictions.argmax(-1)
labels = trainer.predict(test_dataset).label_ids

report = classification_report(labels, preds,
                               target_names=['Clear Reply','Clear Non-Reply','Ambivalent'],
                               digits=4,
                               output_dict=True)  # output_dict=True lets us save it easily

# Print nicely
import json
print(json.dumps(report, indent=4))

import matplotlib.pyplot as plt
from google.colab import files

labels_names = ['Clear Reply','Clear Non-Reply','Ambivalent']
precision = [report[label]['precision'] for label in labels_names]
recall = [report[label]['recall'] for label in labels_names]
f1 = [report[label]['f1-score'] for label in labels_names]

x = range(len(labels_names))
plt.figure(figsize=(6,4))
plt.bar(x, precision, width=0.2, label='Precision', align='center')
plt.bar([i+0.2 for i in x], recall, width=0.2, label='Recall', align='center')
plt.bar([i+0.4 for i in x], f1, width=0.2, label='F1-score', align='center')
plt.xticks([i+0.2 for i in x], labels_names)
plt.ylim(0,1)
plt.title("Clarity Task Metrics per Class")
plt.ylabel("Score")
plt.legend()

# Save the plot
plt.savefig("clarity_metrics_plot.pdf")  # you can also use .png

# Download the plot to your laptop
files.download("clarity_metrics_plot.pdf")

plt.show()

import matplotlib.pyplot as plt

metrics_dict = {'Loss': 0.7639543414115906, 
                'Accuracy': 0.6883116883116883, 
                'F1 Score': 0.6777287402015392}

plt.figure(figsize=(6,4))
plt.bar(metrics_dict.keys(), metrics_dict.values(), color=['red', 'green', 'blue'])
plt.ylim(0,1)  # accuracy and F1 are <1
plt.title("BERT Clarity Task Metrics")
plt.show()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Get predictions
preds = trainer.predict(test_dataset).predictions.argmax(-1)
labels = trainer.predict(test_dataset).label_ids

cm = confusion_matrix(labels, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=['Clear Reply','Clear Non-Reply','Ambivalent'])
plt.figure(figsize=(5,5))
disp.plot(cmap=plt.cm.Blues, values_format='d')  # show numbers
plt.title("Clarity Task Confusion Matrix")
plt.show()





















































