import json
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import pickle
import os

# TODO: Define function that splits sentences that are too long (> max_length) into smaller chunks

# Embedding function
def generate_embeddings(tokenizer, model, sentences, batch_size=32, max_length=256,
                        pooling="mean", exclude_special_tokens=True, to_numpy=True, device=None):

    """Generate mean-pooled sentence embeddings"""

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
            elif pooling == "mean":
                attn = enc["attention_mask"].float()                   # (B, T)
                if exclude_special_tokens and "special_tokens_mask" in enc:
                    attn = attn * (1.0 - enc["special_tokens_mask"].float())
                attn = attn.unsqueeze(-1)                               # (B, T, 1)
                pooled = (h * attn).sum(1) / attn.sum(1).clamp(min=1e-9)
            else:
                raise ValueError(f"Pooling method {pooling} is not yet supported")

        out_chunks.append(pooled.cpu())
        del enc, h, pooled
        torch.cuda.empty_cache()

    out = torch.cat(out_chunks, dim=0)
    return out.numpy() if to_numpy else out

import argparse

def main(args):
    # If the `--regenerate` flag is not set and the file exists, we will not regenerate embeddings
    if not args.regenerate and args.output_dir:
        import os
        # Check if the output directory exists and contains the embeddings file
        if os.path.exists(os.path.join(args.output_dir, "bert_embeddings.pkl")) and os.path.exists(os.path.join(args.output_dir, "roberta_embeddings.pkl")):
            print("Embeddings already exist. Use --regenerate to overwrite.")
            return
        print("Embeddings do not exist or --regenerate flag is set. Proceeding to forced embeddings regeneration...")

    # Define model names

    # Load all tokenizers and models once
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define Tokenizers and Models
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)

    # Initialize the embedding dictionary
    embeddings = {}
    
    # Extract arguments from the command line
    data_file_path = args.data_file
    batch_size = args.batch_size
    max_length = args.max_length
    pooling = args.pooling
    exclude_special_tokens = args.exclude_special_tokens
    to_numpy = args.to_numpy

    # Check if the data file path is provided
    if not data_file_path:
        raise ValueError("Please set the correct path to your data file in the script.")

    print("Generating embeddings...")

    # Load the data
    with open(data_file_path, "r") as f:
        data = json.load(f)
    
    if "sentence" not in data:
        sentence_label = "sentences"
    else:
        sentence_label = "sentence"

    # TODO: Add code to support for multiple models if needed
    # Generate embeddings for each model 
    for movie, characters in tqdm(data[sentence_label].items(), desc="Processing movies"):
        embeddings[movie] = {}

        for character, sentences in tqdm(characters.items(), desc=f"Processing characters in {movie}", leave=False):
            embeddings[movie][character] = generate_embeddings(
                tokenizer, model, sentences, batch_size=batch_size, max_length=max_length,
                pooling=pooling, exclude_special_tokens=exclude_special_tokens, to_numpy=to_numpy, device=device
            )

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Save each embedding dictionary
    with open(os.path.join(output_dir, f"sentence_embeddings_{args.model_name}.pkl"), "wb") as f:
        pickle.dump(embeddings, f)

    print("All embeddings saved.")

if __name__ == "__main__":
    print("Starting embeddings generation script...")

    parser = argparse.ArgumentParser(description="Generate sentence embeddings for characters in movies.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the generated embeddings.")
    parser.add_argument("--data_file", type=str, default="data", help="Path to the data file containing sentences.")
    parser.add_argument("--regenerate", action="store_true", help="Regenerate embeddings even if they already exist.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for embedding generation.")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum length of sentences for tokenization.")
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "cls"], help="Pooling strategy for embeddings.")
    parser.add_argument("--exclude_special_tokens", action="store_true", help="Exclude special tokens from embeddings.")
    parser.add_argument("--to_numpy", action="store_true", help="Return embeddings as numpy arrays instead of tensors.")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Base model name for tokenization.")
    args = parser.parse_args()
    
    # Print the configuration
    print(f"Configuration: {args}")

    main(args)
    print("Embeddings generation completed and saved to disk.")
