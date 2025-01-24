import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import load_dataset
import logging
import os

# Set your desired cache directory
cache_dir = "Y:/huggingface_cache"  # Replace with your preferred path

# Ensure the directory exists
os.makedirs(cache_dir, exist_ok=True)


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def measure_gender_bias(model, tokenizer, test_sentences):
    bias_scores = []
    for sentence in test_sentences:
        male_input = tokenizer(sentence.replace("{GENDER}", "he"), return_tensors="pt")
        female_input = tokenizer(sentence.replace("{GENDER}", "she"), return_tensors="pt")
        
        male_output = model(**male_input).logits
        female_output = model(**female_input).logits
        
        bias_score = torch.abs(male_output.mean() - female_output.mean())
        bias_scores.append(bias_score.item())
    
    return np.mean(bias_scores)

def debias_with_counterfactual_data_augmentation(original_dataset):
    debiased_dataset = []
    for item in original_dataset:
        # Placeholder for actual counterfactual data augmentation
        counterfactual_item = item.copy()
        # Example: Swap gender-related attributes
        if 'gender' in counterfactual_item:
            counterfactual_item['gender'] = 'female' if counterfactual_item['gender'] == 'male' else 'male'
        debiased_dataset.append(counterfactual_item)
    
    return debiased_dataset

class BiasAwareTrainer(Trainer):
    def compute_bias_penalty(self, model):
        # Simplified bias penalty calculation
        bias_penalty = 0
        for name, param in model.named_parameters():
            if 'bias' in name:
                bias_penalty += torch.norm(param)
        return bias_penalty

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss
        
        bias_penalty = self.compute_bias_penalty(model)
        total_loss = loss + 0.1 * bias_penalty
        
        return (total_loss, outputs) if return_outputs else total_loss

def comprehensive_bias_analysis(model, tokenizer):
    # Placeholder test sentences (replace with actual test cases)
    test_sentences = [
        "The {GENDER} doctor was very skilled.",
        "The {GENDER} engineer solved the complex problem.",
        # Add more test sentences covering various scenarios
    ]
    
    bias_tests = {
        "gender_bias": measure_gender_bias(model, tokenizer, test_sentences)
    }
    
    return bias_tests

def main(args):
    try:
        # 1. Load the model and tokenizer
        logger.info(f"Loading model: {args.model}")
        tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=cache_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model, device_map="auto", cache_dir=cache_dir, torch_dtype=torch.float16)

        # 2. Load dataset
        logger.info("Loading dataset")
        original_dataset = load_dataset(args.dataset)

        # 3. Initial bias measurement
        logger.info("Performing initial bias analysis")
        initial_bias_scores = comprehensive_bias_analysis(model, tokenizer)
        logger.info("Initial Bias Scores:")
        for bias_type, score in initial_bias_scores.items():
            logger.info(f"{bias_type}: {score}")

        # 4. Create debiased dataset
        logger.info("Generating counterfactual augmented dataset")
        debiased_dataset = debias_with_counterfactual_data_augmentation(
            original_dataset['train']
        )

        # 5. Prepare training arguments
        training_args = TrainingArguments(
            output_dir="./debiased_model",
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            logging_dir='./logs',
            logging_steps=10,
        )

        # 6. Fine-tune with bias-aware trainer
        logger.info("Starting bias-aware model fine-tuning")
        bias_trainer = BiasAwareTrainer(
            model=model,
            args=training_args,
            train_dataset=debiased_dataset
        )
        
        # Perform training
        bias_trainer.train()

        # 7. Save the debiased model
        debiased_model_path = "./debiased_flan_t5"
        logger.info(f"Saving debiased model to {debiased_model_path}")
        model.save_pretrained(debiased_model_path)
        tokenizer.save_pretrained(debiased_model_path)

        # 8. Final bias measurement
        logger.info("Performing final bias analysis")
        final_bias_scores = comprehensive_bias_analysis(model, tokenizer)
        logger.info("Final Bias Scores:")
        for bias_type, score in final_bias_scores.items():
            logger.info(f"{bias_type}: {score}")

        # 9. Compare and log bias reduction
        logger.info("Bias Reduction Summary:")
        for bias_type in initial_bias_scores:
            reduction = ((initial_bias_scores[bias_type] - final_bias_scores[bias_type]) 
                         / initial_bias_scores[bias_type]) * 100
            logger.info(f"{bias_type} reduction: {reduction:.2f}%")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Bias Mitigation for Large Language Models")
    parser.add_argument(
        "--model", 
        default="google/flan-t5-xxl", 
        help="Hugging Face model name"
    )
    parser.add_argument(
        "--dataset", 
        default="winogender", 
        help="Dataset for bias analysis and fine-tuning"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=3, 
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=4, 
        help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=2e-5, 
        help="Learning rate for fine-tuning"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)