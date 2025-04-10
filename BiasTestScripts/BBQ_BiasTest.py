"""
Script to evaluate a locally saved Hugging Face Causal LM on the BBQ benchmark.

Calculates overall accuracy and disambiguated accuracy.
Includes placeholders for the specific BBQ bias score calculation, which needs
to be implemented based on the original BBQ paper (Parrish et al., 2022)
or reference implementations (e.g., lm-evaluation-harness).

Usage:
  python run_bbq_evaluation.py --model_path /path/to/your/local/model \
                               --output_file /path/to/save/results.json \
                               [--dataset_name Elfsong/BBQ] \
                               [--device cuda]
"""

import os
import json
import argparse
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

# --- Helper Functions ---

def load_bbq_data(dataset_name="Elfsong/BBQ", category=None):
    """Loads a specific category/split of the BBQ dataset."""
    try:
        # Load a specific category (e.g., 'age', 'gender_identity') if provided
        # The 'name' parameter might be needed depending on the dataset structure on Hub
        # Using 'test' split as evaluation is usually done on test sets. Adjust if needed.
        dataset = load_dataset(dataset_name, name=category, split='test', trust_remote_code=True)
        print(f"Loaded BBQ dataset '{dataset_name}', category: {category}, Size: {len(dataset)}")
        # Inspect the dataset features to know the exact column names
        print("Dataset features:", dataset.features)
        # Expected columns might be: 'context', 'question', 'ans0', 'ans1', 'ans2', 'label', 'category', 'context_condition' etc.
        return dataset
    except Exception as e:
        print(f"Error loading BBQ dataset '{dataset_name}' category '{category}': {e}")
        print("Please verify the dataset name/category and availability on Hugging Face Hub.")
        return None

def get_bbq_prediction(model, tokenizer, example, device):
    """
    Gets the model's prediction for a single BBQ example using log likelihood.
    Assumes example has keys like 'context', 'question', 'ans0', 'ans1', 'ans2'.
    Adjust keys based on the actual dataset structure.
    """
    # --- Verify and retrieve necessary keys ---
    required_keys = ['context', 'question', 'ans0', 'ans1', 'ans2']
    if not all(key in example for key in required_keys):
        missing_keys = [key for key in required_keys if key not in example]
        raise KeyError(f"Example missing required keys: {missing_keys}. Found keys: {list(example.keys())}")

    context = example['context']
    question = example['question']
    answer_choices = [example['ans0'], example['ans1'], example['ans2']] # Assuming 3 answer choices.

    # --- Basic prompt structure - Adapt as needed for your specific model ---
    # Some models might need specific separators or instruction formats.
    base_prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"

    log_likelihoods = []
    with torch.no_grad():
        for choice in answer_choices:
            # Ensure choice is a string and add leading space if appropriate
            choice_str = str(choice).strip()
            full_text = base_prompt + " " + choice_str

            # Tokenize and send to device
            inputs = tokenizer(full_text, return_tensors="pt", padding=True, truncation=True, max_length=tokenizer.model_max_length).to(device)
            # Ensure labels are also on the correct device
            labels = inputs["input_ids"].clone()

            # For Causal LMs, typically ignore loss calculation for prompt tokens
            # Find the start index of the answer choice tokens
            # This can be tricky; a simpler approach is using the loss over the whole sequence,
            # assuming the prompt length doesn't vary *too* much between choices.
            # Or calculate log probability of only the answer tokens.
            # Using the model's loss output is the most straightforward way:
            outputs = model(**inputs, labels=labels)

            # Use negative log likelihood (lower is better)
            # The loss is typically the average NLL per token. Multiply by sequence length for total NLL.
            neg_log_likelihood = outputs.loss.item() * inputs.input_ids.shape[1]
            log_likelihoods.append(-neg_log_likelihood) # Store positive log likelihood (higher is better)

    # Prediction is the index of the answer with the highest log likelihood
    prediction_index = np.argmax(log_likelihoods)
    return prediction_index


def evaluate_bbq(model, tokenizer, bbq_dataset, device):
    """
    Evaluates the model on a loaded BBQ dataset split.
    Returns accuracy and bias scores (requires specific BBQ logic).
    """
    model.eval() # Set model to evaluation mode
    predictions = []
    labels = []
    context_conditions = []
    example_ids = [] # Useful for debugging specific examples

    # --- IMPORTANT: Identify the correct column names from your loaded dataset ---
    # These are *guesses* based on common BBQ structures - VERIFY with dataset.features!
    label_col = 'label'
    context_col = 'context_condition' # e.g., 'disambiguated', 'ambiguous'
    id_col = 'example_id' # Or index

    # Check if expected columns exist
    required_eval_keys = [label_col, context_col, id_col]
    if not all(key in bbq_dataset.features for key in required_eval_keys):
        missing = [k for k in required_eval_keys if k not in bbq_dataset.features]
        raise KeyError(f"Dataset missing required columns for evaluation: {missing}. Found: {list(bbq_dataset.features.keys())}")

    for i, example in enumerate(tqdm(bbq_dataset, desc="Evaluating BBQ")):
        try:
            pred_idx = get_bbq_prediction(model, tokenizer, example, device)
            predictions.append(pred_idx)
            labels.append(example[label_col])
            context_conditions.append(example[context_col])
            example_ids.append(example.get(id_col, i)) # Use index if ID column missing
        except Exception as e:
            print(f"Skipping example index {i} (ID: {example.get(id_col, 'N/A')}) due to error: {e}")
            # Append dummy values or handle differently
            predictions.append(-1) # Indicate error
            labels.append(-1)
            context_conditions.append("error")
            example_ids.append(example.get(id_col, i))

    # --- Scoring Logic ---
    predictions = np.array(predictions)
    labels = np.array(labels)
    context_conditions = np.array(context_conditions)

    # Filter out errors
    valid_indices = (predictions != -1)
    if not np.any(valid_indices):
        print("No valid predictions were made.")
        return {"accuracy": 0, "disambiguated_accuracy": 0, "bias_score": "N/A - No valid predictions"}

    # 1. Overall Accuracy
    accuracy = np.mean(predictions[valid_indices] == labels[valid_indices]) * 100

    # 2. Disambiguated Accuracy (Accuracy only on non-ambiguous questions)
    # Find the actual value representing disambiguated contexts (e.g., 'disambiguated', 'disambig')
    # Check unique values: print(np.unique(context_conditions))
    disamb_context_value = 'disambiguated' # ASSUMPTION - VERIFY!
    disamb_indices = valid_indices & (context_conditions == disamb_context_value)
    disamb_accuracy = np.mean(predictions[disamb_indices] == labels[disamb_indices]) * 100 if np.any(disamb_indices) else 0

    # --- 3. Bias Score (PLACEHOLDER - REQUIRES SPECIFIC BBQ SCORING LOGIC) ---
    # This calculation is complex and needs the exact definitions from Parrish et al., 2022.
    # It typically involves analyzing performance on 'ambiguous' examples where the 'label'
    # points to the 'unknown' answer choice, comparing preferences for stereotypical vs.
    # non-stereotypical incorrect answers.
    # Consult the paper or lm-evaluation-harness for the correct implementation.
    bias_score = "Bias Score Calculation Needs Implementation Based on BBQ Paper"
    print("\nWARNING: BBQ Bias Score calculation is complex and requires specific logic from the original paper.")
    print("This script calculates accuracy metrics but provides only a placeholder for the bias score.")
    print("Please consult the BBQ paper (Parrish et al., 2022) or reference implementations.\n")


    results = {
        "accuracy": f"{accuracy:.2f}%",
        "disambiguated_accuracy": f"{disamb_accuracy:.2f}%",
        "bias_score": bias_score, # Replace with actual calculation when implemented
        "total_examples": len(bbq_dataset),
        "valid_predictions": int(np.sum(valid_indices))
    }
    print(f"Evaluation Results:\n{json.dumps(results, indent=2)}")
    return results


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Run BBQ Bias Evaluation on a local Hugging Face model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the locally saved fine-tuned model directory.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the JSON evaluation results.")
    parser.add_argument("--dataset_name", type=str, default="Elfsong/BBQ", help="Name of the BBQ dataset on Hugging Face Hub.")
    parser.add_argument("--device", type=str, default=None, help="Device to run evaluation on (e.g., 'cuda', 'cpu'). Auto-detects if None.")
    parser.add_argument("--categories", nargs='+', default=None, help="Specific BBQ categories to evaluate (e.g., age gender_identity). Evaluates all if None.")

    args = parser.parse_args()

    # --- Setup Device ---
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Model and Tokenizer ---
    print(f"Loading model from: {args.model_path}")
    try:
        # Add trust_remote_code=True if model requires it
        model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        # Ensure pad token is set for tokenizer if needed (often uses eos_token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("Tokenizer pad_token set to eos_token.")
    except Exception as e:
        print(f"Error loading model or tokenizer from {args.model_path}: {e}")
        return

    # --- Define BBQ Categories ---
    # From https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/bbq/README.md
    all_bbq_categories = [
        'age', 'disability_status', 'gender_identity', 'nationality',
        'physical_appearance', 'race_ethnicity', 'religion', 'ses',
        'sexual_orientation',
        # Intersectional categories often require specific handling - check dataset/paper
        # 'race_x_gender', 'race_x_ses'
    ]
    categories_to_evaluate = args.categories if args.categories else all_bbq_categories

    # --- Run Evaluation Loop ---
    all_results = {}
    for category in categories_to_evaluate:
        print(f"\n===== Evaluating BBQ category: {category} =====")
        bbq_data_split = load_bbq_data(dataset_name=args.dataset_name, category=category)
        if bbq_data_split:
            category_results = evaluate_bbq(model, tokenizer, bbq_data_split, device)
            all_results[category] = category_results
        else:
            print(f"Skipping category {category} due to loading error.")
            all_results[category] = {"error": "Failed to load dataset category."}

    # --- Save Results ---
    print(f"\n--- Saving evaluation results to: {args.output_file} ---")
    try:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print("Results saved successfully.")
    except Exception as e:
        print(f"Error saving results to {args.output_file}: {e}")

    print("\n===== BBQ Evaluation Complete =====")

if __name__ == "__main__":
    main()