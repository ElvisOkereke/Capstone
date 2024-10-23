from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
import os
import json

# Set your desired cache directory
cache_dir = "Y:/huggingface_cache"  # Replace with your preferred path

# Ensure the directory exists
os.makedirs(cache_dir, exist_ok=True)

# Use a smaller model (GPT-2)
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)

# Load the dataset
stereoset = load_dataset("stereoset", "intersentence")

# Inspect the data structure
print("Dataset structure:")
print(f"Keys in the dataset: {stereoset.keys()}")
print(f"Types of splits: {[split for split in stereoset]}")
print(f"Number of examples in validation split: {len(stereoset['validation'])}")
print(f"Columns in validation split: {stereoset['validation'].column_names}")

print("\nFirst example:")
first_example = stereoset['validation'][0]
print(json.dumps(first_example, indent=2))

def evaluate_stereoset(model, tokenizer, example):
    context = example['context']
    sentences = example['sentences']
    
    scores = []
    for sentence in sentences:
        inputs = tokenizer(context + " " + sentence, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        score = outputs.logits[:, -1, :].softmax(dim=-1)[0, inputs.input_ids[0, -1]].item()
        scores.append(score)
    
    return {"scores": scores}

# Limit the dataset size
max_examples = 5  # Reduced for initial testing
results = []
total_examples = min(max_examples, len(stereoset['validation']))
print(f"\nProcessing {total_examples} examples")

for i in range(total_examples):
    example = stereoset['validation'][i]
    print(f"\nProcessing example {i+1}/{total_examples}")
    result = evaluate_stereoset(model, tokenizer, example)
    results.append(result)
    print(f"Result: {result}")

# Print detailed results
print("\nDetailed results:")
for i in range(total_examples):
    example = stereoset['validation'][i]
    result = results[i]
    print(f"\nExample {i+1}:")
    print(f"Context: {example['context']}")
    for j, sentence in enumerate(example['sentences']):
        print(f"Sentence {j+1}: {sentence}")
    print(f"Scores: {result['scores']}")
    print(f"Labels: {example['labels']}")
    print(f"Target: {example['target']}")