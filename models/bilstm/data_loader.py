# File: models/bilstm/data_loader.py
"""
Simple loader for the CLARITY QEvasion dataset using HuggingFace `datasets`.
Exports: load_data()
"""
from datasets import load_dataset




def load_data():
    ds = load_dataset("ailsntua/QEvasion")
    train = ds["train"]
    test = ds["test"]
    return train, test




if __name__ == "__main__":
    train, test = load_data()
    print(train)
    print(test)




# ------------------------------------------------------------------------------