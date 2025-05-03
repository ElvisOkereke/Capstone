# Bias Measurement and Mitigation in Large Language Models

## Overview
This repository contains the codebase for the Capstone Design Project titled "Bias Measurement and Mitigation in Large Language Models," conducted by Mehdi Ahsan, Azan Haider, Michael Nguyen, and Elvis Okereke at Toronto Metropolitan University, Department of Electrical, Computer and Biomedical Engineering. The project focuses on detecting, quantifying, and mitigating bias in Large Language Models (LLMs) using a comprehensive framework that includes advanced evaluation techniques and mitigation strategies.

The project introduces a modular pipeline for bias assessment and mitigation, leveraging tools like Gender Logits Score, WEAT/SEAT encoding tests, CrowS-Pairs benchmark, SelfDiagnose, and Perspective API. Mitigation strategies include Counterfactual Data Augmentation (CDA), Self-Debiasing, Social Contact Debiasing (SCD), and the development of MBIAS, an instruction-finetuned model for safety intervention. The codebase supports reproducible research and is designed to be extensible for future bias metrics and techniques.

## Repository Structure
The codebase is organized into several directories, each corresponding to specific components of the project:

- **`data/`**: Contains datasets used for training, validation, and evaluation, including custom datasets for SCD and MBIAS fine-tuning.
- **`models/`**: Includes pre-trained and fine-tuned model checkpoints (e.g., Flan-T5, google/gemma-2b, GPT-2) and configurations for MBIAS.
- **`scripts/`**: Houses Python scripts for bias measurement, mitigation, and evaluation:
  - `Seq2Seq_LogitsLoss_Pipeline.py`: Implements Gender Logits Loss for Flan-T5.
  - `Seq2Seq_SFTSelfReflect_Pipeline.py`: Implements Supervised Fine-Tuning with Self-Reflection for Flan-T5.
  - `SCD_finetuning.py`: Implements Social Contact Debiasing for google/gemma-2b.
  - `self_debias.py`: Implements the Self-Debiasing algorithm for GPT-2 models.
- **`evaluation/`**: Contains scripts for running bias and toxicity evaluations using SEAT, GLD, CrowS-Pairs, BBQ, and Perspective API.
- **`utils/`**: Utility functions for data preprocessing, model loading, and logging.
- **`notebooks/`**: Jupyter notebooks for exploratory analysis and visualization of bias metrics.
- **`docs/`**: Documentation, including detailed descriptions of methods, assumptions, and limitations.

## Installation
To set up the codebase, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ElvisOkereke/Capstone.git
   cd Capstone
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   Install the required Python packages listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

   Key dependencies include:
   - `transformers`: For model loading and fine-tuning.
   - `torch`: For PyTorch-based model training.
   - `datasets`: For handling datasets.
   - `perspective`: For toxicity evaluation via Perspective API.
   - `numpy`, `pandas`: For data manipulation.
   - `matplotlib`, `seaborn`: For visualization.

4. **Download Pre-trained Models**:
   Download pre-trained models (e.g., Flan-T5, google/gemma-2b, GPT-2) from Hugging Face and place them in the `models/` directory. Alternatively, use the provided scripts in `scripts/download_models.py` to automate this process.

5. **Set Up Perspective API**:
   Obtain an API key from [Google Perspective API](https://www.perspectiveapi.com/) and set it as an environment variable:
   ```bash
   export PERSPECTIVE_API_KEY='your_api_key'
   ```

## Usage
The codebase supports bias measurement, mitigation, and evaluation workflows. Below are examples of how to use the main scripts:

### 1. Social Contact Debiasing (SCD)
To fine-tune the google/gemma-2b model with SCD:
```bash
python scripts/SCD_finetuning.py --model_name google/gemma-2b --dataset_path data/scd_dataset.json --output_dir models/gemma_scd
```

### 2. Gender Logits Loss
To apply Gender Logits Loss on Flan-T5:
```bash
python scripts/Seq2Seq_LogitsLoss_Pipeline.py --model_name flan-t5-base --dataset_path data/gender_dataset.json --output_dir models/flan_t5_logits
```

### 3. Self-Debiasing
To apply Self-Debiasing on GPT-2 during text generation:
```bash
python scripts/self_debias.py --model_name gpt2 --prompt "The engineer was working on" --bias_description "gender bias" --output_dir results/self_debias
```

### 4. Evaluation
To evaluate a model for bias and toxicity:
```bash
python evaluation/evaluate_bias.py --model_path models/flan_t5_logits --tests seat,gld,crows_pairs --output_dir results/evaluation
```

### 5. Visualizing Results
To generate plots for SEAT and GLD metrics:
```bash
python notebooks/visualize_results.py --results_dir results/evaluation --output_dir figures/
```

## Key Features
- **Bias Measurement**: Implements Gender Logits Difference (GLD), SEAT, WEAT, CrowS-Pairs, and BBQ for comprehensive bias assessment.
- **Mitigation Strategies**:
  - **Social Contact Debiasing (SCD)**: Uses curated datasets to simulate diverse social interactions, reducing prejudice based on the contact hypothesis.
  - **Gender Logits Loss**: Adjusts model decoding to balance probabilities for gendered terms.
  - **Self-Debiasing**: Modifies token probabilities during generation to avoid biased outputs.
  - **Self-Reflection**: Simulates multi-agent interactions to detect and mitigate implicit biases.
- **Toxicity Evaluation**: Integrates Perspective API for real-time toxicity scoring.
- **Modular Framework**: Allows plug-and-play integration of new datasets, models, and metrics.
- **MBIAS Model**: A custom instruction-finetuned model for enhanced fairness and safety.

## Results
The project achieved significant bias reductions:
- **SCD on google/gemma-2b**: Reduced ability bias by 0.0211 (SEAT effect size), though social and contact biases showed limited improvement.
- **Gender Logits Loss & SFT Self-Reflect on Flan-T5**: Eliminated GLD score (from 0.0038 to 0.0) and reduced SEAT biases (e.g., gender_career from -0.236 to -0.193).
- **Self-Debiasing on GPT-2**: Reduced toxicity scores by up to 46% (Perspective API) across attributes like toxicity and identity attack.

Refer to `results/evaluation/` for detailed metrics and `notebooks/visualize_results.ipynb` for visualizations.

## Limitations
- **Statistical Significance**: Some SEAT results lacked statistical significance (p > 0.05), limiting conclusive claims.
- **Model-Specific Results**: Effectiveness varies across models (e.g., Flan-T5 vs. gemma-2b).
- **Scope**: Limited exploration of multilingual and intersectional biases due to time constraints.
- **Over-Correction**: Self-Debiasing may produce overly neutral outputs, impacting fluency.

## Future Work
- Expand to multilingual and intersectional bias mitigation.
- Enhance SelfDiagnose for better interpretability.
- Integrate real-time adaptive mitigation for dynamic applications.
- Evaluate on larger foundation models for scalability.

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make changes and commit (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request with a detailed description.

Please ensure code follows PEP 8 style guidelines and includes appropriate documentation.

## License
This project is licensed under the MIT License. See `LICENSE` for details.

## Acknowledgements
We thank our Faculty Learning Coordinator, Dr. Ensan, for her guidance and support throughout this project. This work was conducted as part of the Computer/Electrical Engineering Capstone Design Project at Toronto Metropolitan University.

## Citation
If you use this codebase or refer to our work, please cite:
```
Ahsan, M., Haider, A., Nguyen, M., & Okereke, E. (2025). Bias Measurement and Mitigation in Large Language Models. Toronto Metropolitan University Capstone Design Project.
```

## Contact
For questions or inquiries, contact:
- Mehdi Ahsan: mehdi.ahsan@torontomu.ca
- Azan Haider: azan.haider@torontomu.ca
- Michael Nguyen: hiepvan.nguyen@torontomu.ca
- Elvis Okereke: eokereke@torontomu.ca