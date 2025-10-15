import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import random
from tqdm import tqdm

from collections import defaultdict
import torch

from transformers import AutoTokenizer, AutoModel

import gc, os, json, torch
import argparse

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def get_sentence_embeddings(sentences, model, tokenizer, device, batch_size=64, pooling_method = "mean"):
    all_embeddings = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        with torch.no_grad():
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            ).to(device)
            output = model(**encoded)
            attention_mask = encoded["attention_mask"].unsqueeze(-1)
            last_hidden = output.last_hidden_state

            if pooling_method == "mean":
                masked_embeddings = last_hidden * attention_mask
                sum_embeddings = masked_embeddings.sum(dim=1)
                sum_mask = attention_mask.sum(dim=1).clamp(min=1e-9)
                sentence_embeddings = sum_embeddings / sum_mask # mean-pooled sentence embeddings
            elif pooling_method == "cls":
                sentence_embeddings = last_hidden[:, 0, :]
            else:
                raise ValueError(f"Pooling method {pooling_method} is not supported")

            all_embeddings.append(sentence_embeddings.cpu())


            torch.cuda.empty_cache()
            gc.collect()

    return torch.cat(all_embeddings, dim=0)

def data_preprocess(
    model_name,
    source_data_path,
    output_dir,
    threshold=20,                # Threshold represents the minimum number of past sentences each data point should have
    moral_only_past_sentences=False,
    pooling_method="mean",
    sampling_strategy="none",     # "none" | "down" | "up"
    repeat=1,                     # how many balanced splits to create
    seed=42
):
    random.seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_model = AutoModel.from_pretrained(model_name).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    bert_model.eval()

    with open(source_data_path, "r") as f:
        data = json.load(f)

    sentences_data = data["sentence"]
    # sentences_data = data["sentences"]
    ground_truths  = data["ground_truths"]

    all_records = []

    if threshold <= 0:
        print("Threshold should be greater than 0 as each data point should at least have 1 past sentence")

    # Build all records (no split yet)
    for movie, characters in tqdm(sentences_data.items(), desc="Processing characters"):
        for character, sentences in characters.items():
            num_sentences = len(sentences)
            if num_sentences <= threshold:
                continue

            labels = ground_truths[movie][character]

            try:
                sentence_embeddings = get_sentence_embeddings(
                    sentences, bert_model, tokenizer, bert_model.device, pooling_method=pooling_method
                )

                for idx in range(threshold, num_sentences):
                    if moral_only_past_sentences:
                        prior_labels = labels[:idx]
                        moral_mask = torch.tensor(prior_labels, dtype=torch.bool, device=sentence_embeddings.device)
                        if moral_mask.any():
                            past_embeds = sentence_embeddings[:idx][moral_mask]
                            past_sents  = [s for s, y in zip(sentences[:idx], prior_labels) if y == 1]
                        else:
                            past_embeds = sentence_embeddings[:idx]
                            past_sents  = sentences[:idx]
                    else:
                        past_embeds = sentence_embeddings[:idx]
                        past_sents  = sentences[:idx]

                    avg_embedding = past_embeds.mean(dim=0)

                    record = {
                        "movie": movie,
                        "character": character,
                        "sentence": sentences[idx],
                        "label": labels[idx],
                        "avg_embedding": avg_embedding.tolist(),
                        "past_sentences": past_sents,
                        "moral_only_history": bool(moral_only_past_sentences),
                        "history_len": len(past_sents),
                        "history_pos_count": int(sum(labels[:idx])) if moral_only_past_sentences else None,
                    }
                    all_records.append(record)

            except RuntimeError as e:
                print(f"Skipping {character} from {movie} due to memory error: {e}")
            finally:
                torch.cuda.empty_cache()
                gc.collect()

    # Split once into train/test (70/30)
    split_idx = int(len(all_records) * 0.7)
    full_train_data = all_records[:split_idx]
    test_data = all_records[split_idx:]   # keep test set untouched (natural imbalance)

    # Apply repeatable sampling to training set
    os.makedirs(output_dir, exist_ok=True)

    for r in range(repeat):
        if sampling_strategy == "none":
            train_data = full_train_data[:]

        else:
            pos_samples = [rec for rec in full_train_data if rec["label"] == "Yes"]
            neg_samples = [rec for rec in full_train_data if rec["label"] == "No"]

            if sampling_strategy == "down":
                sampled_neg = random.sample(neg_samples, k=len(pos_samples))
                train_data = pos_samples + sampled_neg
            elif sampling_strategy == "up":
                sampled_pos = random.choices(pos_samples, k=len(neg_samples))
                train_data = sampled_pos + neg_samples

            random.shuffle(train_data)

        # Save split
        suffix = pooling_method
        if sampling_strategy == "down":
            suffix += f"_downsampled_split{r+1}"
        elif sampling_strategy == "up":
            suffix += f"_upsampled_split{r+1}"
        else:
            suffix += f"_regular"

        with open(os.path.join(output_dir, f"train_data_{suffix}.json"), "w") as f:
            json.dump(train_data, f)
        if r == 0:  # only need to save test once
            with open(os.path.join(output_dir, f"test_data_{suffix}.json"), "w") as f:
                json.dump(test_data, f)

    print(f"Saved train/test datasets using model: {model_name} (strategy={sampling_strategy}, repeat={repeat})")


def main(args):
    """
    Main function to preprocess data for moral classification training.

    This function serves as the entry point for the data preprocessing pipeline.
    It takes in command-line arguments, extracts the necessary parameters, and
    invokes the `data_preprocess` function to process the data accordingly.

    Args:
        args (argparse.Namespace): A namespace object containing the following attributes:
            - model_name (str): The name of the model to be used for processing.
            - source_data_path (str): The file path to the source data to be processed.
            - output_dir (str): The directory where the processed data will be saved.
            - threshold (float): A threshold value used for filtering or classification.
            - moral_only_past_sentences (bool): A flag indicating whether to include only
              morally relevant past sentences in the processing.
            - pooling_method (str): The method to be used for pooling data (e.g., "mean", "max").

    Returns:
        None

    Raises:
        ValueError: If any of the required arguments are missing or invalid.
        FileNotFoundError: If the source data file does not exist.
        Exception: For any other errors encountered during data preprocessing.

    Example:
        To run the function, use the following command-line arguments:
        ```
        python data_processing.py --model_name "bert-base-uncased" \
                                  --source_data_path "/path/to/source_data.json" \
                                  --output_dir "/path/to/output_dir" \
                                  --threshold 0.5 \
                                  --moral_only_past_sentences True \
                                  --pooling_method "mean"
        ```
    """
    if args.reprocess or not os.path.exists(os.path.join(args.output_dir, f"train_data_{args.pooling_method}.json")):
        if not args.reprocess:
            print(f"Data files not found in {args.output_dir}. Running data preprocessing...")
        else:
            print(f"--reprocess flag set. Regenerating data in {args.output_dir}...")

        data_preprocess(
            model_name=args.model_name,
            source_data_path=args.source_data_path,
            output_dir=args.output_dir,
            threshold=args.threshold,
            moral_only_past_sentences=args.moral_only_past_sentences,
            pooling_method=args.pooling_method,
            sampling_strategy=args.sampling_strategy,
            repeat=args.repeat
        )
    else:
        print(
            f"Data files found in {args.output_dir}. Skipping data preprocessing. "
            f"Use --reprocess to force regeneration.")

if __name__ == "__main__":
    print("Starting data preprocessing for moral relevance classification...")

    parser = argparse.ArgumentParser(description="Preprocess data for moral relevance classification")
    parser.add_argument("--model_name", type=str, required=True, help="Hugging Face model name")
    parser.add_argument("--source_data_path", type=str, required=False, help="Path to the source data JSON file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save processed data")
    parser.add_argument("--threshold", type=int, default=20, help="Minimum number of sentences per character")
    parser.add_argument("--moral_only_past_sentences", action="store_true", help="Use only moral past sentences for training")
    parser.add_argument("--reprocess", action="store_true", help="Reprocess data even if it exists")
    parser.add_argument("--pooling_method", type=str, default="mean", choices=["mean", "cls"], help="Pooling method for sentence embeddings")
    parser.add_argument("--sampling_strategy", type=str, default="none", choices=["none", "down", "up"], help="Sampling strategy for training data")
    parser.add_argument("--repeat", type=int, default=1, help="Number of balanced splits to create")

    args = parser.parse_args()

    main(args)
    print("Data preprocessing completed.")