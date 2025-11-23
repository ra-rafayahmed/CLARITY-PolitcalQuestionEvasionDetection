# CLARITY---Politcal-Question-Evasion-Detection
A research project by Team d-ai-logue focused on analyzing political interview transcripts using NLP to detect clarity, evasion, and communication patterns. The repository includes dataset exploration, label quality analysis, and modeling components for building a multimodal question–answer understanding system.

This repository documents Team d-ai-logue’s participation in the SemEval CLARITY shared task, which focuses on detecting clarity, ambiguity, and evasive techniques in political interview question–answer pairs.

Our work progresses through three main stages:

1. Exploratory Data Analysis (EDA)

(Completed — Assignment 1)

This stage focuses on understanding the dataset and identifying key modeling challenges.
Our EDA includes:

Distribution analysis for clarity and evasion labels

Identification of class imbalance and noisy annotations

Multimodal alignment checks between questions, answers, and metadata

Missing-value visualization

Token-length distributions, sentiment patterns, and cross-feature correlations

All plots exported as PDFs inside the plots/ directory

The EDA provides essential insight into the structure and limitations of the dataset.

2. Benchmark Model Testing (upcoming)

This stage will introduce baseline experiments and initial model comparisons.
Code and results will be added to the benchmarks/ folder as they are completed.

3. Final Model Selection, Training & Evaluation (upcoming)

This stage will focus on training the full model, tuning it, and evaluating performance using the official metrics.
Final implementations will be placed in the final_model/ folder.

Repository Structure
├── eda/
│   ├── eda_notebook.ipynb
│   ├── eda_utils.py
│   └── ...
│
├── plots/
│   ├── clarity_label_distribution.pdf
│   ├── evasion_label_distribution.pdf
│   └── ...
│
├── benchmarks/            # (Will be added later)
│   └── ...
│
├── final_model/           # (Will be added later)
│   └── ...
│
├── data/
│   └── dataset_loading_scripts.py
│
├── src/
│   ├── preprocessing.py
│   ├── modeling_utils.py
│   └── ...
│
└── README.md

