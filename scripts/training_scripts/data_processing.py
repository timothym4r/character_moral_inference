import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import random
from tqdm import tqdm

from collections import defaultdict
import torch
from torch.utils.data import DataLoader, TensorDataset


from transformers import BertTokenizer, BertForMaskedLM, BertModel, AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import re
from torch.nn.functional import normalize

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import gc, os, json, torch


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
    threshold=100,
    moral_only_past_sentences=False,   # <- controls filtering
    pooling_method = "mean"
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_model = AutoModel.from_pretrained(model_name).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    bert_model.eval()

    with open(source_data_path, "r") as f:
        data = json.load(f)

    sentences_data = data["sentences"]
    ground_truths  = data["ground_truths"]

    train_data, test_data = [], []

    for movie, characters in tqdm(sentences_data.items(), desc="Processing characters"):
        for character, sentences in characters.items():
            num_sentences = len(sentences)
            if num_sentences < threshold:
                continue

            labels = ground_truths[movie][character]  # list[int] same length as sentences

            try:
                # Encode all sentences for this character once (batched inside)
                sentence_embeddings = get_sentence_embeddings(
                    sentences, bert_model, tokenizer, bert_model.device, pooling_method = pooling_method
                )  # shape: (num_sentences, H)

                test_start_idx = int(num_sentences * 0.7)

                # iterate over targets; history is sentences[:idx]
                for idx in range(1, num_sentences):
                    # Build mask for prior moral sentences if requested
                    if moral_only_past_sentences:
                        prior_labels = labels[:idx]
                        moral_mask = torch.tensor(prior_labels, dtype=torch.bool, device=sentence_embeddings.device)
                        # Keep only moral=1; if none, fall back to all prior sentences
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

                    if idx < test_start_idx:
                        train_data.append(record)
                    else:
                        test_data.append(record)

            except RuntimeError as e:
                print(f"Skipping {character} from {movie} due to memory error: {e}")

            finally:
                torch.cuda.empty_cache()
                gc.collect()

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, f"train_data_{pooling_method}.json"), "w") as f:
        json.dump(train_data, f)
    with open(os.path.join(output_dir, f"test_data_{pooling_method}.json"), "w") as f:
        json.dump(test_data, f)

    print("Saved train/test datasets using model:", model_name)

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Preprocess data for moral relevance classification")
    parser.add_argument("--model_name", type=str, required=True, help="Hugging Face model name")
    parser.add_argument("--source_data_path", type=str, required=True, help="Path to the source data JSON file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save processed data")
    parser.add_argument("--threshold", type=int, default=100, help="Minimum number of sentences per character")
    parser.add_argument("--moral_only_past_sentences", action="store_true", help="Use only moral past sentences for training")
    parser.add_argument("--pooling_method", type=str, default="mean", choices=["mean", "cls"], help="Pooling method for sentence embeddings")

    args = parser.parse_args()

    data_preprocess(
        model_name=args.model_name,
        source_data_path=args.source_data_path,
        output_dir=args.output_dir,
        threshold=args.threshold,
        moral_only_past_sentences=args.moral_only_past_sentences,
        pooling_method=args.pooling_method
    )
