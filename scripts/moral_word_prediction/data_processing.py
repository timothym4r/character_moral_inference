import os, json, torch, random, gc, argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def get_sentence_embeddings(sentences, model, tokenizer, device, batch_size=64, pooling_method="mean"):
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

            outputs = model(**encoded)
            attention_mask = encoded["attention_mask"].unsqueeze(-1)
            last_hidden = outputs.last_hidden_state

            if pooling_method == "mean":
                masked_embeddings = last_hidden * attention_mask
                sum_embeddings = masked_embeddings.sum(dim=1)
                sum_mask = attention_mask.sum(dim=1).clamp(min=1e-9)
                sentence_embeddings = sum_embeddings / sum_mask
            elif pooling_method == "cls":
                sentence_embeddings = last_hidden[:, 0, :]
            else:
                raise ValueError(f"Unsupported pooling method: {pooling_method}")

            all_embeddings.append(sentence_embeddings.cpu())

        torch.cuda.empty_cache()
        gc.collect()

    return torch.cat(all_embeddings, dim=0)

def data_preprocess(model_name, source_data_path, output_dir, threshold=20, pooling_method="mean", reprocess=False, mask_prediction_index=None):
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, f"train_data_{pooling_method}_{threshold}.json")
    test_path = os.path.join(output_dir, f"test_data_{pooling_method}_{threshold}.json")

    if not reprocess and os.path.exists(train_path):
        print(f"Found existing data in {output_dir}. Use --reprocess to regenerate.")
        return

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    print("Loading source data...")
    with open(source_data_path, "r") as f:
        moral_data = json.load(f)

    moral_dialogue = moral_data["moral_dialogue"]
    moral_dialogue_masked = moral_data["moral_dialogue_masked"]
    ground_truths = moral_data["ground_truths"]

    all_records = []

    for movie, characters in tqdm(moral_dialogue.items(), desc="Processing characters"):
        for character, original_sentences in characters.items():
            num_sentences = len(original_sentences)
            if num_sentences < threshold:
                continue

            try:
                embeddings = get_sentence_embeddings(
                    original_sentences, model, tokenizer, model.device, pooling_method=pooling_method
                )

                masked_sentences = moral_dialogue_masked[movie][character]
                moral_words = ground_truths[movie][character]

                for idx in range(threshold, num_sentences):
                    if mask_prediction_index is not None and mask_prediction_index[movie][character][idx] == 0:
                        continue

                    past_embeds = embeddings[:idx]
                    avg_embedding = past_embeds.mean(dim=0)

                    record = {
                        "movie": movie,
                        "character": character,
                        "masked_sentence": masked_sentences[idx],
                        "target_word": moral_words[idx],
                        "avg_embedding": avg_embedding.tolist(),
                        "past_sentences": original_sentences[:idx],
                        "history_len": idx
                    }
                    all_records.append(record)

            except RuntimeError as e:
                print(f"Skipping {character} from {movie} due to memory error: {e}")
                torch.cuda.empty_cache()
                gc.collect()

    # Split train/test once
    random.shuffle(all_records)
    split_idx = int(0.7 * len(all_records))
    train_data, test_data = all_records[:split_idx], all_records[split_idx:]

    with open(train_path, "w") as f:
        json.dump(train_data, f)
    with open(test_path, "w") as f:
        json.dump(test_data, f)

    print(f"Saved train/test data at {output_dir} ({len(train_data)} train / {len(test_data)} test).")


def main(args):

    mask_prediction_index = None
    if args.mask_prediction_index:
        with open(args.mask_prediction_index, "r") as f:
            mask_prediction_index = json.load(f)

    data_preprocess(
        model_name=args.model_name,
        source_data_path=args.source_data_path,
        output_dir=args.output_dir,
        threshold=args.threshold,
        pooling_method=args.pooling_method,
        reprocess=args.reprocess,
        mask_prediction_index=mask_prediction_index
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Preprocess data for moral word prediction")
    parser.add_argument("--model_name", type=str, required=True, help="Hugging Face model name")
    parser.add_argument("--source_data_path", type=str, required=True, help="Path to source JSON file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save processed data")
    parser.add_argument("--threshold", type=int, default=20, help="Minimum sentences per character")
    parser.add_argument("--pooling_method", type=str, default="mean", choices=["mean", "cls"], help="Pooling method")
    parser.add_argument("--reprocess", action="store_true", help="Force reprocessing even if files exist")
    parser.add_argument("--mask_prediction_index", type=str, default=None, help="Path to mask prediction index JSON file")
    args = parser.parse_args()

    print("Starting preprocessing for moral word prediction...")
    main(args)
    print("Preprocessing completed.")

