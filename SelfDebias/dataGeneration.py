import json
from datasets import load_dataset
from tqdm.auto import tqdm
import sys

def create_jsonl_files(dataset_name="allenai/real-toxicity-prompts",
                       split="train",
                       output_debias_file="SelfDebiasData.jsonl",
                       output_diagnose_file="SelfDiagnoseData.jsonl"):
    """
    Loads the RealToxicityPrompts dataset and creates two JSONL files
    in the specified formats.

    Args:
        dataset_name (str): The name of the dataset on Hugging Face Hub.
        split (str): The dataset split to process (e.g., 'train').
        output_debias_file (str): Path for the SelfDebiasData JSONL file.
        output_diagnose_file (str): Path for the SelfDiagnoseData JSONL file.
    """
    print(f"Loading dataset '{dataset_name}', split '{split}'...")
    try:
        # Load the dataset
        # Using trust_remote_code=True might be necessary for some datasets
        dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
        print(f"Dataset loaded successfully. Number of examples: {len(dataset)}")
        print(f"Dataset features: {dataset.features}")

        # Verify required features based on search results
        required_prompt_keys = ['text', 'toxicity', 'severe_toxicity', 'profanity', 'sexually_explicit', 'flirtation', 'identity_attack', 'threat', 'insult']
        if 'prompt' not in dataset.features or not isinstance(dataset.features['prompt'], dict):
             print(f"Error: Feature 'prompt' not found or not a dictionary in dataset features.", file=sys.stderr)
             return
        if 'challenging' not in dataset.features:
             print(f"Error: Feature 'challenging' not found in dataset features.", file=sys.stderr)
             return
        if not all(key in dataset.features['prompt'] for key in required_prompt_keys):
            missing_keys = [key for key in required_prompt_keys if key not in dataset.features['prompt']]
            print(f"Warning: Dataset 'prompt' feature might be missing expected toxicity keys: {missing_keys}", file=sys.stderr)


    except Exception as e:
        print(f"Error loading dataset '{dataset_name}': {e}", file=sys.stderr)
        print("Please ensure the dataset name is correct and you have internet access.", file=sys.stderr)
        return

    print(f"Processing dataset and writing to:")
    print(f"  1. {output_debias_file}")
    print(f"  2. {output_diagnose_file}")

    try:
        # Open output files
        with open(output_debias_file, 'w', encoding='utf-8') as f_debias, \
             open(output_diagnose_file, 'w', encoding='utf-8') as f_diagnose:

            # Iterate through the dataset with progress bar
            for example in tqdm(dataset, desc="Processing prompts"):
                prompt_data = example.get('prompt')
                challenging_flag = example.get('challenging')

                if prompt_data is None or 'text' not in prompt_data or challenging_flag is None:
                    print(f"Warning: Skipping example due to missing 'prompt', 'prompt.text', or 'challenging' data: {example}", file=sys.stderr)
                    continue

                prompt_text = prompt_data['text']

                # --- Prepare and write SelfDebiasData.jsonl line ---
                debias_record = {
                    "prompt": {"text": prompt_text},
                    "challenging": bool(challenging_flag) # Ensure boolean type
                }
                f_debias.write(json.dumps(debias_record) + '\n')

                # --- Prepare and write SelfDiagnoseData.jsonl line ---
                diagnose_record = {"text": prompt_text}
                # Extract toxicity scores, using None if a key is missing
                diagnose_record["toxicity"] = prompt_data.get("toxicity", None)
                diagnose_record["profanity"] = prompt_data.get("profanity", None)
                diagnose_record["sexually_explicit"] = prompt_data.get("sexually_explicit", None)
                diagnose_record["flirtation"] = prompt_data.get("flirtation", None)
                diagnose_record["identity_attack"] = prompt_data.get("identity_attack", None)
                diagnose_record["threat"] = prompt_data.get("threat", None)
                diagnose_record["insult"] = prompt_data.get("insult", None)
                diagnose_record["severe_toxicity"] = prompt_data.get("severe_toxicity", None)

                f_diagnose.write(json.dumps(diagnose_record) + '\n')

        print("\nProcessing complete.")
        print(f"Successfully created '{output_debias_file}'")
        print(f"Successfully created '{output_diagnose_file}'")

    except IOError as e:
        print(f"Error writing to output file: {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}", file=sys.stderr)


if __name__ == "__main__":
    # You can change the dataset name or output filenames here if needed
    DATASET_ID = "allenai/real-toxicity-prompts" # Identified from search results
    DEBIAS_FILENAME = "SelfDebiasData.jsonl"
    DIAGNOSE_FILENAME = "SelfDiagnoseData.jsonl"

    create_jsonl_files(dataset_name=DATASET_ID,
                       output_debias_file=DEBIAS_FILENAME,
                       output_diagnose_file=DIAGNOSE_FILENAME)