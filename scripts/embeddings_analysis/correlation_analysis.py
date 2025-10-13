import pickle
import os
import numpy as np
import argparse
import torch
import json
from tqdm import tqdm
from embeddings_analysis.utils import get_strong_correlations, plot_r2_scores, find_files_with_key_words

from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load each embedding dictionary

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# Set directory
embedding_dir = "../data/structured_embeddings"

class Autoencoder(nn.Module):
    def __init__(self, input_dim=768, latent_dim=20, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

class MoralClassifier(nn.Module):
    """A classifier that can inject character information into BERT embeddings."""

    def __init__(self, base_model, latent_dim=768, inject_operation = "summation"):
        super().__init__()
        self.bert = base_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 1)  # Binary classification (moral or not)
        self.operation = inject_operation

    def forward(self, input_ids, attention_mask, char_vec=None):
        """
        A forward function. This can be extended to support more operations.
        """

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token

        if char_vec is not None:
            if self.operation == "summation":
                cls_embedding = cls_embedding + char_vec  # Inject character info

        cls_embedding = self.dropout(cls_embedding)
        logits = self.classifier(cls_embedding)
        return logits


def method_1(embeddings_data, ratings_data):
    latent_list, rating_list = [], []
    

    for movie in embeddings_data:
        for character in embeddings_data[movie]:
            latent = embeddings_data[movie][character]
            rating = ratings_data[movie][character]

            if latent is not None and rating is not None:
                latent_list.append(torch.tensor(latent))
                rating_list.append(torch.tensor(rating))

    latent_matrix = torch.stack(latent_list).numpy()
    rating_matrix = torch.stack(rating_list).numpy()

    num_latent = latent_matrix.shape[1]
    num_traits = rating_matrix.shape[1]
    correlation_matrix = np.zeros((num_latent, num_traits))

    for i in range(num_latent):
        for j in range(num_traits):
            x, y = latent_matrix[:, i], rating_matrix[:, j]
            correlation_matrix[i, j] = np.nan if np.std(x) == 0 or np.std(y) == 0 else spearmanr(x, y)[0]

    return pd.DataFrame(
        correlation_matrix,
        index=[f"latent_{i}" for i in range(num_latent)],
        columns=[f"trait_{j}" for j in range(num_traits)]
    )


def method_2(embeddings_data, ratings_data):
    latent_list, rating_list = [], []

    for movie in embeddings_data:
        for character in embeddings_data[movie]:
            latent = embeddings_data[movie][character]
            rating = ratings_data[movie][character]

            if latent is not None and rating is not None:
                latent_list.append(torch.tensor(latent))
                rating_list.append(torch.tensor(rating))

    X = torch.stack(latent_list).numpy()
    Y = torch.stack(rating_list).numpy()

    n_traits = Y.shape[1]
    latent_dim = X.shape[1]
    rows = []

    for j in range(n_traits):
        y = Y[:, j]
        if np.std(y) == 0:
            row = {"trait_index": f"trait_{j}", "r2_score": np.nan}
            row.update({f"latent_{k}": np.nan for k in range(latent_dim)})
        else:
            model = LinearRegression().fit(X, y)
            r2 = r2_score(y, model.predict(X))
            row = {"trait_index": f"trait_{j}", "r2_score": r2}
            row.update({f"latent_{k}": coef for k, coef in enumerate(model.coef_)})
        rows.append(row)

    return pd.DataFrame(rows)

def load_ratings():
    """Load ratings from the specified path.
    
    TODO: Modify the logic to load from original ratings data instead of structured data.
    """

    with open('../../data/dump/structured_data.json', 'r') as f:
        data = json.load(f)

    moral_ratings = {}

    for movie, movie_data in data.items():
        moral_ratings[movie] = {}
        for character, char_data in movie_data["characters"].items():
            if "rating" in char_data:
                moral_ratings[movie][character] = char_data["rating"]
            else:
                moral_ratings[movie][character] = None

    return moral_ratings


def load_model_and_tokenizer(model_name, model_path):
    base_model = AutoModel.from_pretrained(model_name)

    # Reconstruct full classifier model
    classifier = MoralClassifier(base_model)

    # Load state_dict from file
    classifier.load_state_dict(torch.load(model_path, map_location="cpu"))

    # Move to eval mode and device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier.to(device)
    classifier.eval()

    return classifier.bert, AutoTokenizer.from_pretrained(model_name)

def generate_embeddings(tokenizer, model, sentences, batch_size=8, max_length=256,
                        pooling="mean", exclude_special_tokens=True, to_numpy=True, device=None):
  
    """Generate sentence embeddings with a BERT-based model. """
    if device is None:
        device = next(model.parameters()).device
    out_chunks = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        ).to(device)

        with torch.no_grad():
            h = model(**enc).last_hidden_state                         # (B, T, H)

            if pooling == "cls":
                pooled = h[:, 0, :]                                    # (B, H)
            else:
                attn = enc["attention_mask"].float()                   # (B, T)
                if exclude_special_tokens and "special_tokens_mask" in enc:
                    attn = attn * (1.0 - enc["special_tokens_mask"].float())
                attn = attn.unsqueeze(-1)                               # (B, T, 1)
                pooled = (h * attn).sum(1) / attn.sum(1).clamp(min=1e-9)

        out_chunks.append(pooled.cpu())
        del enc, h, pooled
        torch.cuda.empty_cache()

    out = torch.cat(out_chunks, dim=0)
    return out.numpy() if to_numpy else out


def recompute_embeddings(model_path, sentence_data, pooling, batch_size=1, model_name="bert-base-uncased"):
    """Load BERT-based embeddings or recompute them if necessary."""
    
    if pooling not in ["cls", "mean"]:
        raise ValueError("Invalid pooling method. Choose 'cls' or 'mean'.")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name, model_path)

    embeddings_dct = {}

    for movie, characters in tqdm(sentence_data["sentences"].items(), desc="Processing movies"):
        embeddings_dct[movie] = {}

        for character, sentences in tqdm(characters.items(), desc=f"Processing characters in {movie}", leave=False):
            embeddings_dct[movie][character] = generate_embeddings(tokenizer, model, sentences)
    return embeddings_dct 


def encode_latent_embeddings(embeddings, model_H):
    """Encode the averaged character embeddings using the AutoEncoder model."""
    if not isinstance(embeddings, dict):
        raise ValueError("Embeddings should be a dictionary with movie and character keys.")

    encoded_embeddings = {}
    device = next(model_H.parameters()).device  # Ensure tensor is on correct device

    for movie, characters in tqdm(embeddings.items(), desc="Encoding movies"):
        encoded_embeddings[movie] = {}
        for character, sentence_embeddings in tqdm(characters.items(), desc=f"Encoding characters in {movie}", leave=False):
            if len(sentence_embeddings) == 0:  # handle empty list
                print(f"⚠️ No embeddings for {character} in {movie}. Skipping.")
                continue
            sentence_tensor = torch.tensor(sentence_embeddings).float().to(device)  # shape: [N, 768]
            avg_embedding = sentence_tensor.mean(dim=0).unsqueeze(0)               # shape: [1, 768]
            with torch.no_grad():
                _, encoded = model_H(avg_embedding)
            encoded_embeddings[movie][character] = encoded.squeeze().cpu().numpy()  # shape: [latent_dim]

    return encoded_embeddings


def load_or_compute_embeddings(embeddings_path, recompute=False, pooling_method = "cls", model_name = "bert-base-uncased"):
    
    if not recompute:
        try:
            with open(embeddings_path, "rb") as f:
                embeddings = pickle.load(f)
                return embeddings
        except FileNotFoundError:
            try: 
                embeddings_file = os.path.join(embeddings_path, "latent_embeddings.pkl")
                with open(embeddings_file, "rb") as f:
                    embeddings = pickle.load(f)
                    return embeddings
            except FileNotFoundError:
                raise FileNotFoundError("Embeddings file not found. Please recompute embeddings or provide the correct path.")
    else:
        # We need to provide model_name to choose the correct tokenizer and model.
        # Here embeddings_path is a folder containing the model and tokenizer.
        # We need to find the model and tokenizer in the embeddings_path.
        target_folder = embeddings_path
        
        classifier_path = find_files_with_key_words(target_folder, "classifier")

        if classifier_path is None:
            raise FileNotFoundError("Classifier model not found in the specified path.")

        # Define the sentence data
        with open(args.sentence_data_path, "r") as f:
            sentence_data = json.load(f) 

        # embeddings contains the BERT-based embeddings.
        embeddings = recompute_embeddings(classifier_path, sentence_data, pooling_method, model_name = model_name)

        # Save the embeddings to a file
        with open(os.path.join(target_folder, "embeddings.pkl"), "wb") as f:
            pickle.dump(embeddings, f)
        
        print("Embeddings recomputed and saved.")

        # Load the AutoEncoder model (model_H) from the embeddings_path
        ae_path = find_files_with_key_words(target_folder, "model_H")

        if ae_path is None:
            raise FileNotFoundError("AutoEncoder model not found in the specified path.")
        
        model_H = Autoencoder()
        model_H.load_state_dict(torch.load(ae_path, map_location="cpu"))
        model_H.eval()
        model_H.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        latent_embeddings = encode_latent_embeddings(embeddings, model_H)

        # Save the embeddings to a file
        with open(os.path.join(target_folder, "latent_embeddings.pkl"), "wb") as f:
            pickle.dump(latent_embeddings, f)

        print("Latent embeddings computed and saved.")
        
        return latent_embeddings

def run_analysis(embeddings_path, recompute_embeddings = False, pooling_method = "cls", model_name = "bert-base-uncased"):
    """This will load the data required, run the methods, and store the results.

    The results will be stored in the same folder as the embeddings data.
    
    Args:
        embeddings_path (str): Path to the folder containing the embeddings data or the BERT-based model that computes them.
        recompute_embeddings (bool): Whether to recompute embeddings.
    """

    embeddings_data = load_or_compute_embeddings(embeddings_path, recompute=recompute_embeddings, pooling_method=pooling_method, model_name=model_name)
    ratings_data = load_ratings()
    
    # Embeddings data here contains sentence-level embeddings for each character in each movie.
    # Before proceeding to analysis, we need to transform them into character-level emebddings. 
    method_1_result = method_1(embeddings_data, ratings_data)
    method_2_result = method_2(embeddings_data, ratings_data)

    # Save the results
    if recompute_embeddings:
        method_1_result.to_csv(os.path.join(embeddings_path, "method_1_results.csv"))
        method_2_result.to_csv(os.path.join(embeddings_path, "method_2_results.csv"))
        save_folder_path = embeddings_path
    else:
        method_1_result.to_csv(os.path.join(os.path.dirname(embeddings_path), "method_1_results.csv"))
        method_2_result.to_csv(os.path.join(os.path.dirname(embeddings_path), "method_2_results.csv"))
        save_folder_path = Path(embeddings_path).parent


    plot_r2_scores(
        method_2_result,
        top_n=10,
        figsize=(12, 6),
        title="Top 10 Trait-wise R² Scores",
        save_path = os.path.join(save_folder_path, "top_10_r2_scores.png")
    )
    # Get strong correlations
    
    get_strong_correlations(method_1_result, threshold=0.4, save_path=os.path.join(save_folder_path, "strong_correlations.csv"))

    return method_1_result, method_2_result

def main(args):

    run_analysis(
        embeddings_path=args.source_folder_path,
        recompute_embeddings=args.recompute_embeddings,
        model_name=args.model_name,
        pooling_method="cls" if args.cls_embeddings else "mean"
    )

if __name__ == "__main__":
    print("Starting Embeddings analysis...")

    parser = argparse.ArgumentParser(description="Example CLI script")
    
    # Positional argument

    # This needs to be a folder if we want to recompute embeddings and a file if we want to load them.
    # We let the user specify because unlike moral rating data, there are several embedding folders depending
    # on the training trials. 
    parser.add_argument("source_folder_path", help="The path to the source folder")
    # Optional argument with default
    # parser.add_argument("--trainable_base", type=int, default=1, help="Whether the model is trainable or not (1 for yes, 0 for no)")
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-uncased",
        help="The name of the base model to use for embeddings"
    )
    # Flag (boolean switch)
    parser.add_argument("--recompute_embeddings", action="store_true", help="The base model is trainable")
    # Currently we only support CLS and mean-pooled embeddings.
    parser.add_argument("--cls_embeddings", action="store_true", help="Use CLS embeddings for injection")
    parser.add_argument("--sentence_data_path", type=str, help="Path to the sentence data file")

    # Parse the arguments
    args = parser.parse_args()  

    main(args)
    print("Embeddings analysis completed.")