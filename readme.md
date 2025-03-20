# SEQUENCE-TO-SEQUENCE Model Debiasing Framework

A Python-based framework for measuring and mitigating gender and social biases in Open Source language models using multiple debiasing techniques and evaluation metrics.

## Overview

This project implements a comprehensive bias evaluation and mitigation system for T5-based language models, specifically tested on the Flan-T5-base model. It combines multiple approaches to both measure and reduce various types of bias:

1. Direct gender bias measurement using Gender Logit Difference (GLD)
2. Deeper bias evaluation using SEAT (Sentence Encoder Association Test)
3. Custom debiasing loss function for training
4. WinoBias dataset-based evaluation

## Features

- **Multi-metric Bias Evaluation:**
  - Gender Logit Difference (GLD) scoring
  - SEAT testing for multiple bias types:
    - Gender-Career associations
    - Gender-Science associations
    - Race-Sentiment associations
  - WinoBias dataset evaluation

- **Custom Debiasing Components:**
  - WinoBiasDebiasLoss for non-sequential data
  - WinoBiasDebiasLossSeq for sequence generation tasks
  - Stereotype and resolution weight balancing

- **Comprehensive Testing Framework:**
  - Before/After bias evaluation
  - Multiple bias measurement approaches
  - Statistical significance testing

## Requirements

```
torch
transformers
datasets
numpy
scipy
logging
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ElvisOkereke/Capstone
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up your cache directory in the script:
```python
cache_dir = "YOUR_PREFERRED_PATH"
```

## Usage

### Basic Usage

Run the main script:
```bash
python TestFlanT5.py
```

### Custom Configuration

The script includes configurable parameters:

```python
# Debiasing weights
debiasing_loss = WinoBiasDebiasLossSeq(
    stereotype_weight=0.3,
    resolution_weight=0.5
)
```

## How It Works

1. **Initial Bias Measurement:**
   - Measures baseline GLD score
   - Runs SEAT tests for multiple bias types
   - Records initial bias metrics

2. **Debiasing Process:**
   - Uses WinoBias dataset for training
   - Applies custom debiasing loss function
   - Balances between stereotype and resolution losses

3. **Evaluation:**
   - Measures post-training bias metrics
   - Compares before/after results
   - Provides statistical analysis of changes

## Output Interpretation

The script provides detailed logging of bias measurements:

- GLD Scores (closer to 0 is better)
- SEAT Effect Sizes (-1 to 1, where 0 indicates no bias)
- P-values for statistical significance
- Before/After comparisons

Base Scores for Flan-T5:
```
2025-03-20 14:56:50,002 - INFO - GLD Bias Score Before Mitigation: 0.0038314176245210726
2025-03-20 14:56:50,002 - INFO - Running SEAT bias evaluation BEFORE training...
2025-03-20 14:56:50,002 - INFO - Initializing SEAT bias evaluation...
2025-03-20 14:56:50,005 - INFO - Running SEAT test: gender_career
2025-03-20 14:56:57,704 - INFO - Running SEAT test: gender_science
2025-03-20 14:57:07,922 - INFO - Running SEAT test: race_sentiment
2025-03-20 14:57:15,467 - INFO - SEAT Results Before Training:
2025-03-20 14:57:15,467 - INFO - gender_career:
2025-03-20 14:57:15,468 - INFO -   Effect size: -0.236
2025-03-20 14:57:15,468 - INFO -   P-value: 0.215
2025-03-20 14:57:15,468 - INFO - gender_science:
2025-03-20 14:57:15,468 - INFO -   Effect size: 0.102
2025-03-20 14:57:15,468 - INFO -   P-value: 0.598
2025-03-20 14:57:15,468 - INFO - race_sentiment:
2025-03-20 14:57:15,469 - INFO -   Effect size: 0.028
2025-03-20 14:57:15,469 - INFO -   P-value: 0.913
```

Actual Output from Seq2Seq_Logits_Pipeline.py:
```
2025-03-20 14:05:15,752 - INFO - Bias Score After Mitigation: 0.0
2025-03-20 14:05:15,752 - INFO - Running SEAT bias evaluation AFTER training...
2025-03-20 14:05:15,752 - INFO - Initializing SEAT bias evaluation...
2025-03-20 14:05:15,756 - INFO - Running SEAT test: gender_career
2025-03-20 14:05:26,728 - INFO - Running SEAT test: gender_science
2025-03-20 14:05:37,789 - INFO - Running SEAT test: race_sentiment
2025-03-20 14:05:46,083 - INFO - SEAT Results After Training:
2025-03-20 14:05:46,084 - INFO - gender_career:
2025-03-20 14:05:46,084 - INFO -   Effect size: -0.193
2025-03-20 14:05:46,084 - INFO -   P-value: 0.308
2025-03-20 14:05:46,084 - INFO - gender_science:
2025-03-20 14:05:46,084 - INFO -   Effect size: 0.080
2025-03-20 14:05:46,084 - INFO -   P-value: 0.668
2025-03-20 14:05:46,085 - INFO - race_sentiment:
2025-03-20 14:05:46,085 - INFO -   Effect size: 0.000
2025-03-20 14:05:46,085 - INFO -   P-value: 1.000
```

Actual Output from Seq2Seq_SelfReflect_Pipeline.py:
```
2025-03-20 14:05:15,752 - INFO - Bias Score After Mitigation: 0.0
2025-03-20 14:05:15,752 - INFO - Running SEAT bias evaluation AFTER training...
2025-03-20 14:05:15,752 - INFO - Initializing SEAT bias evaluation...
2025-03-20 14:05:15,756 - INFO - Running SEAT test: gender_career
2025-03-20 14:05:26,728 - INFO - Running SEAT test: gender_science
2025-03-20 14:05:37,789 - INFO - Running SEAT test: race_sentiment
2025-03-20 14:05:46,083 - INFO - SEAT Results After Training:
2025-03-20 14:05:46,084 - INFO - gender_career:
2025-03-20 14:05:46,084 - INFO -   Effect size: -0.193
2025-03-20 14:05:46,084 - INFO -   P-value: 0.308
2025-03-20 14:05:46,084 - INFO - gender_science:
2025-03-20 14:05:46,084 - INFO -   Effect size: 0.080
2025-03-20 14:05:46,084 - INFO -   P-value: 0.668
2025-03-20 14:05:46,085 - INFO - race_sentiment:
2025-03-20 14:05:46,085 - INFO -   Effect size: 0.000
2025-03-20 14:05:46,085 - INFO -   P-value: 1.000
```

## Citation

If you use this code in your research, please cite:

- https://arxiv.org/html/2403.14409v1#bib.bib38 [LSDM (Least Square Debias Method)]
- https://arxiv.org/html/2402.11190v1 [Gender Logits Difference (GLD)]
- https://arxiv.org/abs/1807.11714 [CDA]

## Acknowledgments

- WinoBias dataset creators
- Hugging Face team for the T5 implementation
- [Other acknowledgments]
