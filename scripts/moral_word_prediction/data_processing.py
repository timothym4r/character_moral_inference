# Import Libraries and Tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import random
import re
import os
from tqdm import tqdm
from collections import defaultdict

from torch.optim import AdamW
import torch
from torch.utils.data import DataLoader, TensorDataset

# Language Models
from transformers import BertTokenizer, BertModel

# Classification Models
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import argparse

random.seed(42)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)

bert_model.eval()

# Function for data pre-processing

def data_preprocess(source_data_path, output_dir, threshold=20):
    with open(source_data_path, "r") as f:
        moral_data = json.load(f)

    moral_dialogue = moral_data["moral_dialogue"]
    moral_dialogue_masked = moral_data["moral_dialogue_masked"]
    ground_truths = moral_data["ground_truths"]

    train_data_100, test_data_100 = [], []

    for movie, characters in tqdm(moral_dialogue.items(), desc="Processing characters"):
        for character, original_sentences in characters.items():
            num_sentences = len(original_sentences)

            # To make sure each character has enough sentences before later indexing
            if num_sentences < threshold:
                continue

            masked_sentences = moral_dialogue_masked[movie][character]
            moral_words = ground_truths[movie][character]

            # Step 1: Encode all sentences once
            with torch.no_grad():
                encoded = tokenizer(
                    original_sentences,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256
                ).to(device)

                output = bert_model(**encoded)
                sentence_embeddings = output.last_hidden_state.mean(dim=1).cpu()  # shape: [num_sentences, 768]

            # second_half_start = num_sentences // 2
            second_half_start = threshold

            char_data_rows = []

            for idx in range(second_half_start, num_sentences):
                if idx == 0:
                    continue  # can't average anything before index 0

                # Step 2: Use already-computed embeddings
                past_embeds = sentence_embeddings[:idx]
                avg_embedding = past_embeds.mean(dim=0)

                record = {
                    "character": character,
                    "avg_embedding": avg_embedding.tolist(),
                    "past_sentences": original_sentences[:idx],
                    "masked_sentence": masked_sentences[idx],
                    "target_word": moral_words[idx],
                    # Keep track of the movie
                    "movie": movie
                }

                char_data_rows.append(record)

            random.shuffle(char_data_rows)

            test_start_idx = int(0.7 * len(char_data_rows))

            rows_for_training = char_data_rows[:test_start_idx]
            rows_for_testing = char_data_rows[test_start_idx:]

            train_data_100.extend(rows_for_training)
            test_data_100.extend(rows_for_testing)

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, f"train_data_{threshold}.json"), "w") as f:
        json.dump(train_data_100, f)

    with open(os.path.join(output_dir, f"test_data_{threshold}.json"), "w") as f:
        json.dump(test_data_100, f)

    print("Saved train/test datasets with optimized embedding reuse.")


def main():
    """
    Main function to preprocess data for moral word prediction.

    This function serves as the entry point for the data preprocessing pipeline.
    It takes in command-line arguments, extracts the necessary parameters, and
    invokes the `data_preprocess` function to process the data accordingly.

    Args:
        None

    Returns:
        None

    Example:
        To run the function, use the following command-line arguments:
        ```
        python data_processing.py --source_data_path "/path/to/source_data.json" \
                                    --output_dir "/path/to/output_dir" \
                                    --threshold 20
        ```
    """
    # TODO: Add more arguments:
    # - model_name (for embeddings)
    # - pooling_method (mean, cls, etc)
    # - reprocess (bool) to skip processing if files exist
    parser = argparse.ArgumentParser(description="Preprocess data for moral word prediction")
    parser.add_argument("--source_data_path", type=str, required=True, help="Path to the source data JSON file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save processed data")
    parser.add_argument("--threshold", type=int, default=20, help="Minimum number of sentences per character")

    args = parser.parse_args()

    data_preprocess(
        source_data_path=args.source_data_path,
        output_dir=args.output_dir,
        threshold=args.threshold
    )

if __name__ == "__main__":
    main()