# dataloader.py

from datasets import load_dataset

def load_qevasion_dataset():
    """
    Loads the QEvasion dataset and returns raw splits.
    No tokenization or preprocessing is done here.
    """
    print("Loading QEvasion dataset...")
    dataset = load_dataset("ailsntua/QEvasion")

    train_data = dataset["train"]
    test_data  = dataset["test"]

    print(f"Train samples: {len(train_data)}")
    print(f"Test samples:  {len(test_data)}")

    return train_data, test_data


if __name__ == "__main__":
    train_data, test_data = load_qevasion_dataset()

    # Show example row structure
    print("\nExample row:\n")
    print(train_data[0])
