import torch
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import List, Dict, Tuple, Optional
import random
from tqdm import tqdm

class LSDMDebias:
    """
    Implementation of the Least Square Debias Method (LSDM) for Seq2Seq models.
    This class implements the approach described in the paper to mitigate gender bias
    in language models by modifying specific weights in the model.
    """
    
    def __init__(
        self,
        model_name: str,
        layers_to_modify: List[int] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the LSDM debias method.
        
        Args:
            model_name: The name of the model to load from HuggingFace
            layers_to_modify: List of layer indices to modify (if None, will determine automatically)
            device: Device to use for computation
        """
        self.device = device
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Default layers to modify if not specified
        self.layers_to_modify = layers_to_modify if layers_to_modify is not None else [3, 4, 5, 6, 7, 8]
        
        # Dictionary to store original and debiased weights
        self.original_weights = {}
        self.debiased_weights = {}
        
        # Placeholders for bias vectors
        self.E = None  # Matrix of male occupation embeddings
        self.P = None  # Matrix of female occupation embeddings
        self.V_star = None  # Target unbiased vector
        self.delta = None  # Weight update
        
    def _get_proj_layer_weights(self, layer_idx: int) -> torch.Tensor:
        """
        Get the projection layer weights for a specific layer.
        Adapts to different model architectures by finding the right parameter.
        
        Args:
            layer_idx: The index of the layer to extract weights from
            
        Returns:
            The weight matrix for the specified layer
        """
        # This needs to be adapted based on the specific model architecture
        # For T5-like models:
        if hasattr(self.model, "encoder"):
            try:
                # Try to access encoder layers for T5
                return self.model.encoder.block[layer_idx].layer[1].DenseReluDense.wo.weight
            except (AttributeError, IndexError):
                pass
                
            try:
                # Try to access decoder layers for T5
                return self.model.decoder.block[layer_idx].layer[1].DenseReluDense.wo.weight
            except (AttributeError, IndexError):
                pass
                
        # For BART-like models:
        if hasattr(self.model, "model"):
            try:
                return self.model.model.encoder.layers[layer_idx].fc2.weight
            except (AttributeError, IndexError):
                pass
                
            try:
                return self.model.model.decoder.layers[layer_idx].fc2.weight
            except (AttributeError, IndexError):
                pass
                
        raise ValueError(f"Could not find projection layer at index {layer_idx} for this model architecture")
            
    def _set_proj_layer_weights(self, layer_idx: int, new_weights: torch.Tensor):
        """
        Set the projection layer weights for a specific layer.
        
        Args:
            layer_idx: The index of the layer to modify
            new_weights: The new weights to set
        """
        # For T5-like models:
        if hasattr(self.model, "encoder"):
            try:
                # Try to access encoder layers for T5
                self.model.encoder.block[layer_idx].layer[1].DenseReluDense.wo.weight.data = new_weights
                return
            except (AttributeError, IndexError):
                pass
                
            try:
                # Try to access decoder layers for T5
                self.model.decoder.block[layer_idx].layer[1].DenseReluDense.wo.weight.data = new_weights
                return
            except (AttributeError, IndexError):
                pass
                
        # For BART-like models:
        if hasattr(self.model, "model"):
            try:
                self.model.model.encoder.layers[layer_idx].fc2.weight.data = new_weights
                return
            except (AttributeError, IndexError):
                pass
                
            try:
                self.model.model.decoder.layers[layer_idx].fc2.weight.data = new_weights
                return
            except (AttributeError, IndexError):
                pass
                
        raise ValueError(f"Could not set projection layer weights at index {layer_idx} for this model architecture")
    
    def generate_bias_sentences(
        self,
        occupation_dataset: Dict[str, List[str]],
        num_sentences: int = 20,
        max_sentence_length: int = 30,
        templates: List[str] = None,
        d_threshold: float = 0.9,
        top_sentences: int = 5
    ) -> Dict[str, List[str]]:
        """
        Generate biased sentences for occupations using the model's generation capabilities.
        
        Args:
            occupation_dataset: Dictionary mapping male and female occupations
            num_sentences: Number of sentences to generate per occupation
            max_sentence_length: Maximum length of generated sentences
            templates: List of template prompts to use (if None, uses default)
            d_threshold: Perplexity threshold for sentence selection
            top_sentences: Number of top sentences to select
            
        Returns:
            Dictionary of occupation -> list of biased sentences
        """
        if templates is None:
            templates = [
                "The {occupation} said that",
                "I met a {occupation} who",
                "The {occupation} is someone who",
                "Working as a {occupation}, they",
                "When I asked the {occupation}, they"
            ]
        
        biased_sentences = {}
        
        # Combine male and female occupations into one set
        all_occupations = list(occupation_dataset.get("male", [])) + list(occupation_dataset.get("female", []))
        
        # Process each occupation
        for occupation in tqdm(all_occupations, desc="Generating biased sentences"):
            occupation_sentences = []
            
            # Generate multiple sentences for each occupation
            for _ in range(num_sentences):
                # Randomly select a template
                template = random.choice(templates)
                prompt = template.format(occupation=occupation)
                
                # Generate text completion
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                
                # For seq2seq models like T5, we need to setup decoder_input_ids
                decoder_input_ids = self.tokenizer(
                    "", return_tensors="pt", add_special_tokens=False
                ).input_ids.to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        decoder_input_ids=decoder_input_ids if decoder_input_ids.shape[1] > 0 else None,
                        max_length=max_sentence_length,
                        num_return_sequences=1,
                        do_sample=True,
                        temperature=0.7
                    )
                    
                # Decode the generated text
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                occupation_sentences.append(generated_text)
            
            # Calculate perplexity for each sentence (simplified for seq2seq models)
            perplexities = []
            for sentence in occupation_sentences:
                inputs = self.tokenizer(sentence, return_tensors="pt").to(self.device)
                labels = inputs["input_ids"].clone()
                
                # Create decoder_input_ids for seq2seq models
                decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels)
                
                with torch.no_grad():
                    outputs = self.model(
                        **inputs,
                        decoder_input_ids=decoder_input_ids,
                        labels=labels
                    )
                    perplexity = torch.exp(outputs.loss).item()
                    perplexities.append(perplexity)
            
            # Select top sentences with highest perplexity
            sorted_sentences = [s for _, s in sorted(zip(perplexities, occupation_sentences), reverse=True)]
            biased_sentences[occupation] = sorted_sentences[:top_sentences]
            
        return biased_sentences
    
    def compute_vectors(
        self,
        occupation_dataset: Dict[str, List[str]],
        biased_sentences: Dict[str, List[str]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the E (male) and P (female) vectors by processing biased sentences.
        
        Args:
            occupation_dataset: Dictionary of occupation pronouns
            biased_sentences: Dictionary of biased sentences for each occupation
            
        Returns:
            Tuple of (E, P) tensors representing male and female occupation embeddings
        """
        male_occupations = occupation_dataset.get("male", [])
        female_occupations = occupation_dataset.get("female", [])
        
        # Initialize empty lists to store vectors
        male_vectors = []
        female_vectors = []
        
        # Process each occupation and compute its vector representation k
        for occupation in tqdm(list(biased_sentences.keys()), desc="Computing occupation vectors"):
            # Get sentences for this occupation
            sentences = biased_sentences[occupation]
            
            # Skip if no sentences
            if not sentences:
                continue
                
            # Process each sentence to get occupation vector representation
            occupation_vectors = []
            for sentence in sentences:
                # Tokenize the sentence
                tokens = self.tokenizer(sentence, return_tensors="pt").to(self.device)
                
                # For seq2seq models, we need to provide decoder input for encoder-decoder attention
                decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(tokens["input_ids"])
                
                # Find the occupation token indices in the sentence
                occupation_tokens = self.tokenizer(occupation, add_special_tokens=False)["input_ids"]
                
                # Get model activations - note the necessary decoder inputs for seq2seq models
                with torch.no_grad():
                    outputs = self.model(
                        **tokens,
                        decoder_input_ids=decoder_input_ids,
                        output_hidden_states=True,
                        return_dict=True
                    )
                    
                # Get the encoder's last hidden state - this is what we need for LSDM
                hidden_states = outputs.encoder_last_hidden_state if hasattr(outputs, "encoder_last_hidden_state") else outputs.last_hidden_state
                
                # Find occupation token positions in the sentence (simplified)
                occupation_positions = []
                input_ids = tokens["input_ids"][0].tolist()
                for i in range(len(input_ids) - len(occupation_tokens) + 1):
                    if input_ids[i:i+len(occupation_tokens)] == occupation_tokens:
                        occupation_positions.extend(list(range(i, i+len(occupation_tokens))))
                
                # If occupation token is found, extract its representation
                if occupation_positions:
                    # Average the hidden states at occupation token positions
                    occupation_vector = torch.mean(
                        hidden_states[0, occupation_positions, :], dim=0
                    )
                    occupation_vectors.append(occupation_vector)
            
            # Average the vectors across sentences
            if occupation_vectors:
                avg_vector = torch.mean(torch.stack(occupation_vectors), dim=0)
                
                # Add to male or female vectors based on occupation category
                if occupation in male_occupations:
                    male_vectors.append(avg_vector)
                elif occupation in female_occupations:
                    female_vectors.append(avg_vector)
        
        # Stack vectors to create E and P matrices
        E = torch.stack(male_vectors).t() if male_vectors else None
        P = torch.stack(female_vectors).t() if female_vectors else None
        
        if E is None or P is None:
            raise ValueError("Failed to compute E or P vectors. Check your occupation dataset.")
            
        return E, P
    
    def optimize_v_star(self, E: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        """
        Optimize to find V* (the unbiased target vector).
        
        Args:
            E: Matrix of male occupation embeddings
            P: Matrix of female occupation embeddings
            
        Returns:
            V*: The target unbiased vector
        """
        # Implementation of Equation 15 and 16 to find v*
        
        # Dimension of the embedding space
        embedding_dim = E.shape[0]
        
        # Initialize v* randomly
        v_star = torch.randn(embedding_dim, device=self.device, requires_grad=True)
        
        # Optimize using gradient descent
        optimizer = torch.optim.Adam([v_star], lr=0.01)
        
        # Number of optimization steps
        num_steps = 1000
        
        for step in tqdm(range(num_steps), desc="Optimizing V*"):
            optimizer.zero_grad()
            
            # Calculate loss based on Equation 16
            loss = 0
            
            # Process male occupations (E)
            for i in range(E.shape[1]):
                e_i = E[:, i]
                prob = torch.sigmoid(torch.dot(v_star, e_i))
                loss -= torch.log(prob)
                
            # Process female occupations (P)
            for i in range(P.shape[1]):
                p_i = P[:, i]
                prob = torch.sigmoid(torch.dot(v_star, p_i))
                loss -= torch.log(1 - prob)
                
            # Average the loss
            loss /= (E.shape[1] + P.shape[1])
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
        return v_star.detach()
    
    def compute_layer_updates(
        self,
        layer_idx: int,
        E: torch.Tensor,
        P: torch.Tensor,
        V_star: torch.Tensor,
        total_layers: int
    ) -> torch.Tensor:
        """
        Compute the update for a specific layer using the LSDM method.
        
        Args:
            layer_idx: Index of the layer to update
            E: Matrix of male occupation embeddings
            P: Matrix of female occupation embeddings
            V_star: Target unbiased vector
            total_layers: Total number of layers to modify
            
        Returns:
            Updated weights for the layer
        """
        # Get current weights of the layer
        W_l = self._get_proj_layer_weights(layer_idx)
        self.original_weights[layer_idx] = W_l.clone()
        
        # The issue is here. E appears to be transposed incorrectly.
        # E should have shape [embedding_dim, num_male_occupations]
        # P should have shape [embedding_dim, num_female_occupations]
        # V_star should have shape [embedding_dim]
        
        # Ensure V_star is a column vector if needed
        if V_star.dim() == 1:
            V_star = V_star.unsqueeze(0)
        
        # Check if E and P are properly oriented
        # E and P should have embedding vectors as columns
        if E.shape[0] != W_l.shape[1]:
            # If E has embeddings as rows instead of columns, transpose it
            E = E.t()
        
        if P.shape[0] != W_l.shape[1]:
            # If P has embeddings as rows instead of columns, transpose it
            P = P.t()
        
        # Calculate E @ E^T and P @ P^T
        EET = E @ E.t()
        PPT = P @ P.t()
        
        # Calculate V* @ E^T and V1 @ P^T
        # Ensure V_star has the right shape for multiplication
        print(f"V_star shape: {V_star.shape}, E shape: {E.shape}, P shape: {P.shape}")
        V_star_ET = V_star.reshape(-1, 1).T @ E
        
        V1_PT = torch.zeros_like(V_star_ET)  # This is a placeholder; in practice, V1 would be computed
        
        # Calculate delta according to Equation 12
        try:
            inverse_term = torch.inverse(EET + PPT)
        except RuntimeError:
            # Add small diagonal term for numerical stability if inverse fails
            print(f"Adding regularization for numerical stability in layer {layer_idx}")
            inverse_term = torch.inverse(EET + PPT + 1e-5 * torch.eye(
                EET.shape[0], device=EET.device
            ))
            
        delta_term = (V_star_ET - W_l @ EET) @ inverse_term
        
        # Implement progressive adjustment according to Equation 18
        l_i = layer_idx
        l_f = total_layers
        
        # Calculate m_l^r according to Equation 18
        factor = l_i / (l_f + 1)
        # Ensure m_l_r has the right shape
        if V_star.dim() > 1 and V_star.shape[0] == 1:
            m_l_r = factor * V_star.squeeze(0)
        else:
            m_l_r = factor * V_star
        
        # Update the weights
        delta = delta_term.clone()
        W_l_new = W_l + delta
        
        self.debiased_weights[layer_idx] = W_l_new
        return W_l_new
    
    def debias(
        self,
        occupation_dataset: Dict[str, List[str]],
        biased_sentences: Optional[Dict[str, List[str]]] = None,
        save_path: Optional[str] = None
    ):
        """
        Apply the LSDM debias method to the model.
        
        Args:
            occupation_dataset: Dictionary of male/female occupations
            biased_sentences: Pre-generated biased sentences (if None, will generate them)
            save_path: Path to save the debiased model (if None, won't save)
        """
        # Step 1: Generate biased sentences if not provided
        if biased_sentences is None:
            print("Generating biased sentences...")
            biased_sentences = self.generate_bias_sentences(occupation_dataset)
        
        # Step 2: Compute E and P vectors
        print("Computing E and P vectors...")
        self.E, self.P = self.compute_vectors(occupation_dataset, biased_sentences)
        
        # Step 3: Optimize V*
        print("Optimizing V*...")
        self.V_star = self.optimize_v_star(self.E, self.P)
        
        # Step 4: Compute and apply updates to each layer
        print("Applying LSDM updates to model weights...")
        total_layers = len(self.layers_to_modify)
        
        for layer_idx in tqdm(self.layers_to_modify, desc="Updating layers"):
            new_weights = self.compute_layer_updates(
                layer_idx,
                self.E,
                self.P,
                self.V_star,
                total_layers
            )
            self._set_proj_layer_weights(layer_idx, new_weights)
        
        # Save the debiased model if requested
        if save_path:
            print(f"Saving debiased model to {save_path}...")
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            
        print("Debiasing complete!")
        
    def reset_model(self):
        """
        Reset the model to its original weights before debiasing.
        """
        for layer_idx, original_weight in self.original_weights.items():
            self._set_proj_layer_weights(layer_idx, original_weight)
        print("Model reset to original weights!")
        
    def evaluate_bias(
        self,
        test_sentences: List[str],
        male_terms: List[str],
        female_terms: List[str]
    ) -> Dict:
        """
        Evaluate the model bias on test sentences.
        
        Args:
            test_sentences: List of test sentences
            male_terms: List of male terms to track
            female_terms: List of female terms to track
            
        Returns:
            Dictionary with bias metrics
        """
        male_count = 0
        female_count = 0
        
        for sentence in tqdm(test_sentences, desc="Evaluating bias"):
            # Tokenize the input
            inputs = self.tokenizer(sentence, return_tensors="pt").to(self.device)
            
            # For seq2seq models, we need to provide decoder input for generation
            decoder_input_ids = self.tokenizer(
                "", return_tensors="pt", add_special_tokens=False
            ).input_ids.to(self.device)
            
            # Generate text
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    decoder_input_ids=decoder_input_ids if decoder_input_ids.shape[1] > 0 else None,
                    max_length=50,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7
                )
                
            # Decode the generated text
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Count occurrences of male and female terms
            for term in male_terms:
                male_count += generated_text.lower().count(term.lower())
            
            for term in female_terms:
                female_count += generated_text.lower().count(term.lower())
        
        total = male_count + female_count
        male_percentage = (male_count / total * 100) if total > 0 else 0
        female_percentage = (female_count / total * 100) if total > 0 else 0
        
        return {
            "male_count": male_count,
            "female_count": female_count,
            "male_percentage": male_percentage,
            "female_percentage": female_percentage,
            "bias_score": abs(male_percentage - female_percentage)
        }


# Example usage
if __name__ == "__main__":
    # Sample dataset of occupations
    occupation_dataset = {
        "male": ["doctor", "engineer", "programmer", "lawyer", "scientist"],
        "female": ["nurse", "teacher", "secretary", "librarian", "assistant"]
    }
    
    # Initialize the LSDM debias method
    lsdm = LSDMDebias(model_name="t5-small", layers_to_modify=[3, 4, 5, 6])
    
    # Generate biased sentences manually (to avoid issues in automated generation)
    biased_sentences = {
        "doctor": [
            "The doctor said that he needs to check the patient's vitals.",
            "I met a doctor who specializes in cardiology.",
            "The doctor is someone who graduated from medical school with honors."
        ],
        "engineer": [
            "The engineer designed a new bridge that can withstand earthquakes.",
            "I met an engineer who works at SpaceX.",
            "The engineer is someone who solves complex problems."
        ],
        # Add more pre-generated sentences for each occupation
        "nurse": [
            "The nurse helped the patient take their medication.",
            "I met a nurse who has been working in the ICU for ten years.",
            "The nurse is someone who provides compassionate care."
        ],
        "teacher": [
            "The teacher prepared lesson plans for the week.",
            "I met a teacher who loves working with children.",
            "The teacher is someone who inspires their students."
        ]
    }
    
    # Apply debiasing with pre-generated sentences
    lsdm.debias(occupation_dataset, biased_sentences=biased_sentences, save_path="debiased_t5_small")
    
    # Evaluate bias before and after debiasing
    test_sentences = [
        "The person works as a doctor and",
        "The professional with a degree in engineering",
        "Someone who teaches children",
        "A skilled individual who provides healthcare"
    ]
    
    male_terms = ["he", "him", "his", "himself", "man", "men", "boy", "boys"]
    female_terms = ["she", "her", "hers", "herself", "woman", "women", "girl", "girls"]
    
    # Reset to original model and evaluate
    lsdm.reset_model()
    original_bias = lsdm.evaluate_bias(test_sentences, male_terms, female_terms)
    print("Original model bias:", original_bias)
    
    # Reapply debiasing and evaluate
    lsdm.debias(occupation_dataset, biased_sentences=biased_sentences)
    debiased_bias = lsdm.evaluate_bias(test_sentences, male_terms, female_terms)
    print("Debiased model bias:", debiased_bias)