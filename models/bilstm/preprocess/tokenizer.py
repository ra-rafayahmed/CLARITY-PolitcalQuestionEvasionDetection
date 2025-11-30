"""
Tokenizer and preprocessing utilities (no TensorFlow dependency).
- build_vocab(texts, vocab_size)
- texts_to_sequences(texts, vocab, unk_token)
- pad_sequences(sequences, max_len)

We use a whitespace + basic punctuation tokenizer to keep things simple and portable.
"""
import re
import json
from collections import Counter
from typing import List, Dict


WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def simple_tokenize(text: str) -> List[str]:
    if text is None:
        return []
    text = text.lower()
    return WORD_RE.findall(text)


def build_vocab(texts: List[str], vocab_size: int = 30000, min_freq: int = 1, special_tokens=("<pad>", "<unk>")) -> Dict[str, int]:
    counter = Counter()
    for t in texts:
        tokens = simple_tokenize(t)
        counter.update(tokens)

    # most common
    most_common = [w for w, c in counter.most_common(vocab_size) if c >= min_freq]

    # build map with special tokens at start
    idx = 0
    vocab = {}
    for s in special_tokens:
        vocab[s] = idx
        idx += 1

    for w in most_common:
        if w in vocab:
            continue
        vocab[w] = idx
        idx += 1

    return vocab


def texts_to_sequences(texts: List[str], vocab: Dict[str, int], unk_token: str = "<unk>") -> List[List[int]]:
    unk_idx = vocab.get(unk_token)
    sequences = []
    for t in texts:
        tokens = simple_tokenize(t)
        seq = [vocab.get(tok, unk_idx) for tok in tokens]
        sequences.append(seq)
    return sequences


def pad_sequences(sequences: List[List[int]], max_len: int, padding: str = "post") -> List[List[int]]:
    padded = []
    for seq in sequences:
        if len(seq) >= max_len:
            padded.append(seq[:max_len])
        else:
            if padding == "post":
                padded.append(seq + [0] * (max_len - len(seq)))
            else:
                padded.append([0] * (max_len - len(seq)) + seq)
    return padded


def save_vocab(vocab: Dict[str, int], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


def load_vocab(path: str) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

