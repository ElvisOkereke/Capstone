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
python Seq2Seq_Model_Test.py
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

Example output:
```
GLD Bias Score Before Mitigation: 0.00383
GLD Bias Score After Mitigation: 5.84e-08

SEAT Results:
gender_career:
  Effect size: -0.236
  P-value: 0.217
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
