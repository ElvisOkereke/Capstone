import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
import logging
import os
import datasets
import json
import math
import random

# Set your desired cache directory
cache_dir = "Y:/huggingface_cache"  # Replace with your preferred path

# Ensure the directory exists
os.makedirs(cache_dir, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

D_values = [8,9,10,11] #Sentence lengths
F_Sentences = 20 #Total number of sentences per occupation
Top_500_Sentences = 500 #Top 500 sentences per perplexity


def get_profession_list():
    file_path = r"Y:\huggingface_cache\jsons\sorted_professions.json"
    with open(file_path, "r") as f:
         professions = [entry[0] for entry in json.load(f)]
    return professions

templates = [
        "The {} said that",
        "The {} yelled that",
        "The {} whispered that",
        "The {} wanted that",
        "The {} desired that",
        "The {} wished that",
        "The {} ate because",
        "The {} ran because",
        "The {} drove because",
        "The {} slept because",
        "The {} cried because",
        "The {} laughed because",
        "The {} went home because",
        "The {} stayed up because",
        "The {} was fired because",
        "The {} was promoted because",
        "The {} yelled because",
    ]
    
def generate_sentences(profession, template, model, tokenizer, device):
    prompt = template.format(profession)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    max_length = random.choice(D_values)

    with torch.no_grad():
        output = model.generate(**inputs, max_length=max_length)

    generated_sentence = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_sentence 


def compute_perplexity(sentences, model, tokenizer, device):
    inputs = tokenizer(sentences, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    
    loss = outputs.loss
    ppl = math.exp(loss.item())  # Compute perplexity
    return ppl

def loss_function(logits, tokenizer):
    """
    Compute the absolute difference in probabilities between "he" and "she".
    """
    he_token = tokenizer.convert_tokens_to_ids("he")
    she_token = tokenizer.convert_tokens_to_ids("she")

    probs = torch.softmax(logits, dim=-1)

    # Ensure tensor has the correct dimensions
    if probs.shape[-1] <= max(he_token, she_token):
        raise ValueError(f"Token index {max(he_token, she_token)} is out of bounds for logits shape {probs.shape}")

    he_prob = probs[0, he_token].item()
    she_prob = probs[0, she_token].item()

    return abs(he_prob - she_prob)

def measure_gender_bias_with_dataset(model, tokenizer, dataset_split):
    """
    Measures Gender Logits Difference (GLD) using the WinoBias dataset.
    """
    bias_scores = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define gendered word pairs
    gender_pairs = [("he", "she"), ("him", "her"), ("his", "hers"), ("man", "woman"),
                    ("boy", "girl"), ("father", "mother"), ("brother", "sister")]

    for row in dataset_split:
        sentence = row["input"]
        tokens = sentence.split()

        # Find gendered words in the sentence
        gendered_indices = [i for i, token in enumerate(tokens) if token in sum(gender_pairs, ())]

        if not gendered_indices:
            continue  # Skip if no gendered words are found

        for male_word, female_word in gender_pairs:
            if male_word in tokens:
                male_sentence = sentence.replace(male_word, male_word)
                female_sentence = sentence.replace(male_word, female_word)

                # Tokenize both versions of the sentence
                male_input = tokenizer(male_sentence, return_tensors="pt", padding=True, truncation=True).to(device)
                female_input = tokenizer(female_sentence, return_tensors="pt", padding=True, truncation=True).to(device)

                # Create decoder input IDs (needed for Seq2Seq models)
                decoder_input_ids = torch.full(
                    (male_input.input_ids.shape[0], 1), 
                    model.config.decoder_start_token_id, 
                    device=device
                )

                # Generate logits for both sentences
                with torch.no_grad():
                    male_logits = model(input_ids=male_input.input_ids, decoder_input_ids=decoder_input_ids).logits
                    female_logits = model(input_ids=female_input.input_ids, decoder_input_ids=decoder_input_ids).logits

                # Compute probabilities
                male_probs = torch.softmax(male_logits, dim=-1).mean().item()
                female_probs = torch.softmax(female_logits, dim=-1).mean().item()

                # Apply GLD formula
                gld_score = abs(male_probs - female_probs) / (male_probs + female_probs)
                bias_scores.append(gld_score)

    return np.mean(bias_scores) if bias_scores else 0.0

def compute_gender_bias_probability(sentences, model, tokenizer, device):
    inputs = tokenizer(sentences, return_tensors="pt").to(device)

    decoder_input_ids = torch.full(
        (inputs["input_ids"].shape[0], 1), tokenizer.pad_token_id, dtype=torch.long
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs, decoder_input_ids=decoder_input_ids)
        logits = outputs.logits[:, -1, :]

    return loss_function(logits, tokenizer)

def select_most_biased_sentences(model, tokenizer, device, file_path):
    """
    Select the 5 sentences with the highest gender bias probability.
    """

    output_file_bias = os.path.join(os.path.dirname(file_path), "top_5_gender_biased_sentences.json")

    with open(file_path, "r", encoding="utf-8") as f:
        top_500_sentences = json.load(f)

    biased_sentences = []

    for entry in top_500_sentences:
        gb_prob = compute_gender_bias_probability(entry["sentence"], model, tokenizer, device)
        entry["P(gb)"] = gb_prob  # Add gender bias probability
        biased_sentences.append(entry)

    # Sort by highest gender bias probability
    top_5_bias_sentences = sorted(biased_sentences, key=lambda x: x["P(gb)"], reverse=True)[:5]

    # Save results
    with open(output_file_bias, "w", encoding="utf-8") as f:
        json.dump(top_5_bias_sentences, f, indent=4)

    print("Top 5 gender-biased sentences saved.")

    return top_5_bias_sentences

def compute_layerwise_bias(model, tokenizer, device, male_sentences, female_sentences):
    layer_bias = {}

    def hook_fn(layer_idx):
        def hook(module, input, output):
            activations_dict[layer_idx] = output.detach().mean(dim=0)  # Store mean activation
        return hook

    for layer_idx in range(len(model.decoder.block)):
        activations_dict = {}

        # Register hooks on all decoder MLP layers
        hooks = []
        for l in range(len(model.decoder.block)):
            handle = model.decoder.block[l].layer[1].register_forward_hook(hook_fn(l))
            hooks.append(handle)

        male_activations = []
        female_activations = []

        # Compute activations for male sentences
        for sentence in male_sentences:
            inputs = tokenizer(sentence, return_tensors="pt").to(device)
            with torch.no_grad():
                _ = model(**inputs)
            male_activations.append(activations_dict[layer_idx])

        # Compute activations for female sentences
        for sentence in female_sentences:
            inputs = tokenizer(sentence, return_tensors="pt").to(device)
            with torch.no_grad():
                _ = model(**inputs)
            female_activations.append(activations_dict[layer_idx])

        # Remove hooks after extraction
        for handle in hooks:
            handle.remove()

        # Compute absolute difference between male & female activations
        bias_score = torch.mean(torch.abs(torch.stack(male_activations) - torch.stack(female_activations)))

        layer_bias[layer_idx] = bias_score.item()  # Store bias intensity for each layer

    return layer_bias

# Extract activations from a specific layer and compute the mean activation
def get_layer_activation(model, tokenizer, device, sentences, layer_num):
    """
    Extracts the mean activation from the DenseReluDense layer of T5 decoder.
    """
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(device)

    decoder_input_ids = torch.full(
        (inputs["input_ids"].shape[0], 1), tokenizer.pad_token_id, dtype=torch.long
    ).to(device)

    activation_storage = []

    def hook_fn(module, input, output):
        if isinstance(output, tuple):  
            output = output[0]  
        activation_storage.append(output.detach())

    # Hook at the DenseReluDense layer (MLP)
    handle = model.decoder.block[layer_num].layer[-1].register_forward_hook(hook_fn)

    with torch.no_grad():
        _ = model(**inputs, decoder_input_ids=decoder_input_ids)

    handle.remove()  # Remove hook after inference

    if not activation_storage:
        raise RuntimeError(f"❌ No activations captured from layer {layer_num}.")

    # Compute Mean Activation
    mean_activation = torch.mean(torch.stack(activation_storage), dim=0)

    return mean_activation


#Function finds the least biased activiation v* by minimizing the loss
def optimize_v_star(model, tokenizer, device, X):
    """
    Optimize v* by minimizing the loss function across all biased sentences.
    """
    activation_storage = []  # Store activations for all 5 sentences

    for entry in X:
        input_text = entry["sentence"]
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)

        decoder_input_ids = torch.full(
            (inputs["input_ids"].shape[0], 1), tokenizer.pad_token_id, dtype=torch.long
        ).to(device)

        def hook_fn(module, input, output):
            if isinstance(output, tuple):  
                output = output[0]  
            activation_storage.append(output.detach())

        # Fix: Attach hook at the decoder's MLP layer
        handle = model.decoder.block[0].layer[-1].register_forward_hook(hook_fn)

        with torch.no_grad():
            _ = model(**inputs, decoder_input_ids=decoder_input_ids)

        handle.remove()  

    if not activation_storage:
        raise RuntimeError("❌ No activations were captured.")

    # Fix: Compute mean across biased sentences (should be shape `[1, hidden_dim]`)
    v_star = torch.mean(torch.stack(activation_storage), dim=0)  

    return v_star

def compute_r_star(v_star, m_hat_i_le):
    """
    Compute the correction vector r* ensuring shape consistency.
    """
    print(f"🔍 Before Fix: v_star {v_star.shape}, m_hat_i_le {m_hat_i_le.shape}")

    # Ensure v_star and m_hat_i_le are (1, hidden_dim)
    v_star = v_star.mean(dim=0, keepdim=True)
    m_hat_i_le = m_hat_i_le.mean(dim=0, keepdim=True)

    print(f"After Fix: v_star {v_star.shape}, m_hat_i_le {m_hat_i_le.shape}")

    return v_star - m_hat_i_le  # Element-wise subtraction

def compute_covariance(activations):
    #Computes covariance matrix P P^T for activations
    P = torch.stack(activations).squeeze(1)  # Stack activations into a matrix
    covariance_matrix = P @ P.mT  # Compute PP^T
    return covariance_matrix

def compute_kernel(X, kernel_function, tokenizer, device):
    """
    Compute kernel values using a kernel function k(s_j + x).
    """
    kernel_values = []

    # Extract only the sentences
    sentences = [entry["sentence"] for entry in X]

    # Convert sentences to tensor embeddings
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        sentence_embeddings = inputs["input_ids"].float()  # Convert token IDs to float tensors

    for i, x in enumerate(sentence_embeddings):
        k_sum = sum(kernel_function(torch.norm(s - x)) for s in sentence_embeddings) / len(sentence_embeddings)
        kernel_values.append(k_sum.item())  # Convert tensor to Python float

    return torch.tensor(kernel_values)


def construct_matrices(E_l, V_star):
    """
    Construct E^l and V* matrices based on the activations.
    """
    E_l_matrix = torch.stack(E_l)  # Convert list of activations into matrix
    V_star_matrix = torch.stack(V_star)  # Convert list of v* into matrix
    return E_l_matrix, V_star_matrix

def construct_matrices(E_l, V_star):
    """
    Construct E^l and V* matrices based on the activations.
    """
    # Ensure `E_l` is a list of tensors
    if isinstance(E_l, torch.Tensor):
        E_l = [E_l]  # Convert to a list

    # Ensure `V_star` is a list of tensors
    if isinstance(V_star, torch.Tensor):
        V_star = [V_star]  # Convert to a list

    # Stack tensors to create matrices
    E_l_matrix = torch.stack(E_l)  # Shape: [num_sentences, 1, 1, 768]
    V_star_matrix = torch.stack(V_star)  # Shape: [1, 1, 1, 768]

    # Ensure correct shape for matrix operations
    E_l_matrix = E_l_matrix.squeeze(2).squeeze(1)  # Now [num_sentences, 768]
    V_star_matrix = V_star_matrix.squeeze(2).squeeze(1)  # Now [1, 768]

    print(f"Fixed Shapes: E_l {E_l_matrix.shape}, V_star {V_star_matrix.shape}")

    return E_l_matrix, V_star_matrix

def compute_weight_update(E_l, V_star, target_shape):
    """
    Compute the least-squares correction term Δ^l using Equation (12) and ensure it matches the shape of T5 weights.
    """
    #Convert tensors to `float32`
    E_l = E_l.to(torch.float32)
    V_star = V_star.to(torch.float32)

    # Expand `V_star` to match `E_l` batch size
    if V_star.shape[0] == 1 and E_l.shape[0] > 1:
        V_star = V_star.expand(E_l.shape[0], -1)  # Expands to [5, 768]

    # Compute pseudo-inverse (for least-squares solution)
    pseudo_inverse = torch.linalg.pinv(E_l @ E_l.T)  

    # Compute Δ^l
    Delta_l = (V_star @ E_l.T) @ pseudo_inverse  # Shape: [5, 5]

    # Fix Shape Mismatch**: Ensure `Delta_l` is correctly reshaped
    hidden_dim, intermediate_dim = target_shape  # `[2048, 768]`
    Delta_l = torch.nn.functional.interpolate(Delta_l.unsqueeze(0).unsqueeze(0), size=(hidden_dim, intermediate_dim), mode="bilinear").squeeze()

    print(f"Fixed Shape: V_star {V_star.shape}, E_l {E_l.shape}, Δ^l {Delta_l.shape}")

    return Delta_l

def apply_weight_update(model, Delta_l, layer_idx):
    """
    Apply the computed weight update Δ^l to the model.
    """
    with torch.no_grad():
        #🔹 Access the correct weight inside the MLP layer (`DenseReluDense` part)
        mlp_layer = model.decoder.block[layer_idx].layer[-1]  # `T5LayerFF`
        
        # T5 MLP layers contain `DenseReluDense`, which has `wi` and `wo` weights
        mlp_layer.DenseReluDense.wo.weight += Delta_l  # Update the output projection weights

    print(f"Applied Weight Update to Layer {layer_idx}")

def perform_LSDM_update(model, tokenizer, device, X, r_star, layer_num):
    """
    Perform LSDM update based on extracted activations and computed r*.
    """
    # Step 1: Compute mean activation
    m_hat_i_le = get_layer_activation(model, tokenizer, device, [x["sentence"] for x in X], layer_num)
    
    # Step 2: Compute covariance matrix P P^T
    activations = [get_layer_activation(model, tokenizer, device, [x["sentence"]], layer_num) for x in X]
    PP_T = compute_covariance(activations)
    
    # Step 3: Compute kernel function values
    kernel_values = compute_kernel(X, lambda x: torch.exp(-torch.norm(x)**2), tokenizer, device)

    # Step 4: Construct matrices E^l and V*
    E_l_matrix, V_star_matrix = construct_matrices(activations, r_star)

    # Fix: Get the correct shape from the model’s weight matrix
    target_shape = model.decoder.block[layer_num].layer[-1].DenseReluDense.wo.weight.shape  # `[2048, 768]`

    # Step 5: Compute weight update Δ^l
    Delta_l = compute_weight_update(E_l_matrix, V_star_matrix, target_shape)

    # Step 6: Apply weight update to model
    apply_weight_update(model, Delta_l, layer_num)

    return model  # Return updated model

def main():
    try:
        # Load model and tokenizer
        logger.info("Loading model: google/flan-t5-base")
        model_name = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, device_map="auto", cache_dir=cache_dir, torch_dtype=torch.bfloat16  
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Define file paths
        output_file = r"Y:\huggingface_cache\jsons\generated_sentences_test.json"
        output_file_top500 = r"Y:\huggingface_cache\jsons\top_500_sentences.json"
        output_file_bias = r"Y:\huggingface_cache\jsons\top_5_gender_biased_sentences.json"

        logger.info("Loading WinoBias dataset")
        ds = load_dataset("Elfsong/Wino_Bias", cache_dir=cache_dir)
        logger.info(f"Available dataset splits: {list(ds.keys())}")

        train_dataset = ds["train"]
        eval_dataset = ds["test"]

        # Step 1: Load generated sentences if they exist
        if os.path.exists(output_file):
            print("Loading existing generated sentences...")
            with open(output_file, "r", encoding="utf-8") as f:
                generated_sentences = json.load(f)
        else:
            print("No existing sentences found. Generating new ones...")
            professions = get_profession_list()
            generated_sentences = []

            for i, profession in enumerate(professions):  
                for _ in range(F_Sentences):  
                    template = random.choice(templates)  
                    sentence = generate_sentences(profession, template, model, tokenizer, device)
                    ppl = compute_perplexity(sentence, model, tokenizer, device)

                    generated_sentences.append({
                        "profession": profession,
                        "template": template,
                        "sentence": sentence,
                        "perplexity": ppl
                    })

                if (i + 1) % 10 == 0:
                    print(f"Generated {F_Sentences} sentences each for {i+1}/{len(professions)} professions.")

            # Save newly generated sentences
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(generated_sentences, f, indent=4)
            print(f"Saved generated sentences to {output_file}")

        # Step 2: Sort sentences by perplexity and keep top 500
        if os.path.exists(output_file_top500):
            print("Loading existing top 500 sentences...")
            with open(output_file_top500, "r", encoding="utf-8") as f:
                top_500_sentences = json.load(f)
        else:
            print("Computing top 500 sentences...")
            top_500_sentences = sorted(generated_sentences, key=lambda x: x["perplexity"], reverse=True)[:Top_500_Sentences]
            with open(output_file_top500, "w", encoding="utf-8") as f:
                json.dump(top_500_sentences, f, indent=4)
            print("Saved top 500 highest perplexity sentences.")

        # Step 3: Compute P(gb) for top 500
        if os.path.exists(output_file_bias):
            print("Loading existing biased sentences...")
            with open(output_file_bias, "r", encoding="utf-8") as f:
                X = json.load(f)
        else:
            print("Computing most biased sentences...")
            X = select_most_biased_sentences(model, tokenizer, device, output_file_top500)

        # Step 4: Compute debiasing components
        layer_num = 0 # Bottom MLP layer
        m_hat_i_le = get_layer_activation(model, tokenizer, device, [x["sentence"] for x in X], layer_num=0)
        v_star = optimize_v_star(model, tokenizer, device, X)
        r_star = compute_r_star(v_star, m_hat_i_le)

        logger.info("Evaluating bias BEFORE LSDM debiasing...")
        bias_score_before = measure_gender_bias_with_dataset(model, tokenizer, eval_dataset)
        logger.info(f"Bias Score Before LSDM: {bias_score_before}")

        model = perform_LSDM_update(model, tokenizer, device, X, r_star, layer_num)

        logger.info("Evaluating bias AFTER LSDM debiasing...")
        bias_score_after = measure_gender_bias_with_dataset(model, tokenizer, eval_dataset)
        logger.info(f"Bias Score After LSDM: {bias_score_after}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

