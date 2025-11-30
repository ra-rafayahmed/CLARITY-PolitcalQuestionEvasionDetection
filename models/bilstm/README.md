"""
Bi-LSTM baseline for CLARITY QEvasion dataset

Usage:
1. Create a Python virtual environment and install requirements from requirements.txt
2. From the project root run:
   python models/bilstm/training/train.py

Results are written to models/bilstm/results/

Notes:
- This script attempts to download GloVe 6B embeddings automatically. If your environment is offline,
  download GloVe manually from http://nlp.stanford.edu/data/glove.6B.zip and place the extracted files in models/bilstm/glove/
"""

