from datasets import load_dataset
from collections import Counter
import re
from nltk import ngrams
import matplotlib.pyplot as plt
import os

# Make folder for plots if it doesn't exist
if not os.path.exists("plots"):
    os.makedirs("plots")

# Step 1: Load the dataset
dataset = load_dataset("ailsntua/QEvasion")

# Step 2: Get all answers
answers = dataset['train']['interview_answer']

# Check first 3 answers
print("First 3 answers:", answers[:3])

# Step 3: Compute vocabulary size
def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

all_tokens = []
for ans in answers:
    all_tokens.extend(tokenize(ans))

vocab_size = len(set(all_tokens))
print("Vocabulary size:", vocab_size)

# Step 4: Find frequent 1-grams and 2-grams
one_grams = Counter(all_tokens)
top_1grams = one_grams.most_common(10)
print("Top 10 words:", top_1grams)

two_grams = Counter(ngrams(all_tokens, 2))
top_2grams = two_grams.most_common(10)
print("Top 10 2-grams:", top_2grams)

# Step 5: Plot top 1-grams
words, counts = zip(*top_1grams)
plt.figure(figsize=(10,5))
plt.bar(words, counts)
plt.title("Top 10 Words in Answers")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots/top_1grams.pdf")
plt.show()

# Step 6: Plot top 2-grams
phrases, counts = zip(*top_2grams)
phrases = [' '.join(p) for p in phrases]
plt.figure(figsize=(10,5))
plt.bar(phrases, counts)
plt.title("Top 10 2-grams in Answers")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots/top_2grams.pdf")
plt.show()
