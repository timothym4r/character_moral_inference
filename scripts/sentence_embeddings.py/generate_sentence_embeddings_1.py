import json
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


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
    model_names_dictionary = {
        "bert": "bert-base-uncased",
        "roberta": "roberta-base",
        "deberta": "microsoft/deberta-v3-base",
        "electra": "google/electra-base-discriminator"
    }

    # Load all tokenizers and models once
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define Tokenizers and Models
    tokenizers = {name: AutoTokenizer.from_pretrained(model_name) for name, model_name in model_names_dictionary.items()}
    models = {name: AutoModel.from_pretrained(model_name).to(device).eval() for name, model_name in model_names_dictionary.items()}

    # Initialize your 4 embedding dictionaries
    bert_embeddings = {}
    roberta_embeddings = {}
    deberta_embeddings = {}
    electra_embeddings = {}
    
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

    # Generate embeddings for each model
    for movie, characters in tqdm(data["sentences"].items(), desc="Processing movies"):
        bert_embeddings[movie] = {}
        roberta_embeddings[movie] = {}
        deberta_embeddings[movie] = {}
        electra_embeddings[movie] = {}

        for character, sentences in tqdm(characters.items(), desc=f"Processing characters in {movie}", leave=False):
            bert_embeddings[movie][character] = generate_embeddings(
                tokenizers["bert"], models["bert"], sentences, batch_size=batch_size, max_length=max_length,
                pooling=pooling, exclude_special_tokens=exclude_special_tokens, to_numpy=to_numpy, device=device
            )
            roberta_embeddings[movie][character] = generate_embeddings(
                tokenizers["roberta"], models["roberta"], sentences, batch_size=batch_size, max_length=max_length,
                pooling=pooling, exclude_special_tokens=exclude_special_tokens, to_numpy=to_numpy, device=device
            )
            deberta_embeddings[movie][character] = generate_embeddings(
                tokenizers["deberta"], models["deberta"], sentences, batch_size=batch_size, max_length=max_length,
                pooling=pooling, exclude_special_tokens=exclude_special_tokens, to_numpy=to_numpy, device=device
            )
            electra_embeddings[movie][character] = generate_embeddings(
                tokenizers["electra"], models["electra"], sentences, batch_size=batch_size, max_length=max_length,
                pooling=pooling, exclude_special_tokens=exclude_special_tokens, to_numpy=to_numpy, device=device
            )

    import pickle
    import os

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Save each embedding dictionary
    with open(os.path.join(output_dir, "bert_embeddings.pkl"), "wb") as f:
        pickle.dump(bert_embeddings, f)

    with open(os.path.join(output_dir, "roberta_embeddings.pkl"), "wb") as f:
        pickle.dump(roberta_embeddings, f)

    with open(os.path.join(output_dir, "deberta_embeddings.pkl"), "wb") as f:
        pickle.dump(deberta_embeddings, f)

    with open(os.path.join(output_dir, "electra_embeddings.pkl"), "wb") as f:
        pickle.dump(electra_embeddings, f)

    print("All embeddings saved.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate sentence embeddings for characters in movies.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the generated embeddings.")
    parser.add_argument("--data_file", type=str, default="data", help="Path to the data file containing sentences.")
    parser.add_argument("--regenerate", action="store_true", help="Regenerate embeddings even if they already exist.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for embedding generation.")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum length of sentences for tokenization.")
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "cls"], help="Pooling strategy for embeddings.")
    parser.add_argument("--exclude_special_tokens", action="store_true", help="Exclude special tokens from embeddings.")
    parser.add_argument("--to_numpy", action="store_true", help="Return embeddings as numpy arrays instead of tensors.")
    args = parser.parse_args()
    
    # Print the configuration
    print(f"Configuration: {args}")

    main(args)
    print("Embeddings generation completed and saved to disk.")

