import os, json, torch, random, gc, argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

TYPE_TOKENS = {
    "spoken": "[SPK]",
    "action": "[ACT]",
}

def prefix_by_type(sentence: str, stype: str) -> str:
    tok = TYPE_TOKENS.get(stype, "[SPK]")
    return f"{tok} {sentence}"

def ensure_special_tokens(tokenizer, model, special_tokens):
    """Add special tokens to tokenizer + resize model embeddings if needed."""
    to_add = [t for t in special_tokens if t not in tokenizer.get_vocab()]
    if len(to_add) > 0:
        tokenizer.add_special_tokens({"additional_special_tokens": to_add})
        model.resize_token_embeddings(len(tokenizer))


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

def data_preprocess(model_name, source_data_path, output_dir, threshold=20, 
                    pooling_method="mean", reprocess=False, sentence_mask_type = None,
                    moving_avg = False, moving_avg_window = -1 # moving_avg_window = -1 means we don't use windowed moving average
                    ):
    
    os.makedirs(output_dir, exist_ok=True)

    if sentence_mask_type is not None:
        train_path = os.path.join(output_dir, f"train_data_{pooling_method}_{threshold}_{sentence_mask_type}.json")  # sentence_mask_type can be "moral_word" or "all"
        test_path = os.path.join(output_dir, f"test_data_{pooling_method}_{threshold}_{sentence_mask_type}.json")
    else:
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
    sentence_type = moral_data["sentence_type"] # "spoken" or "action"

    if sentence_mask_type is not None:
        mask_prediction_index = moral_data["moral_label"]   # Moral label is only given to sentences containing moral word and classified as moral

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
                    if sentence_mask_type is not None and mask_prediction_index[movie][character][idx] == "No":
                        continue
                    
                    # Take past spoken and action sentences separately
                    action_sentences = original_sentences[:idx][sentence_type[movie][character][:idx] == "action"]
                    spoken_sentences = original_sentences[:idx][sentence_type[movie][character][:idx] == "spoken"]

                    action_embeddings = embeddings[:idx][sentence_type[movie][character][:idx] == "action"]
                    spoken_embeddings = embeddings[:idx][sentence_type[movie][character][:idx] == "spoken"]

                    # TODO: Implement the embedding making

                    past_embeds = embeddings[:idx]
                    # If we directly take the mean of empty tensor, it will be NaN and raise an error later
                    if past_embeds.size(0) == 0:
                        avg_embedding = torch.zeros(embeddings.size(1))
                    else:
                        avg_embedding = past_embeds.mean(dim=0)

                    record = {
                        "movie": movie,
                        "character": character,
                        "masked_sentence": masked_sentences[idx],
                        "target_word": moral_words[idx],
                        "avg_embedding": avg_embedding.tolist(),
                        # "past_sentences": original_sentences[:idx],

                        # Now instead of only spoken sentences, we provide all past sentences (spoken + action)
                        "spoken_sentences": spoken_sentences,
                        "action_sentences": action_sentences,
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

    data_preprocess(
        model_name=args.model_name,
        source_data_path=args.source_data_path,
        output_dir=args.output_dir,
        threshold=args.threshold,
        pooling_method=args.pooling_method,
        reprocess=args.reprocess,
        sentence_mask_type=args.sentence_mask_type
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data for moral word prediction")
    parser.add_argument("--model_name", type=str, required=True, help="Hugging Face model name")
    parser.add_argument("--source_data_path", type=str, required=True, help="Path to source JSON file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save processed data")
    parser.add_argument("--threshold", type=int, default=20, help="Minimum sentences per character")
    parser.add_argument("--pooling_method", type=str, default="mean", choices=["mean", "cls"], help="Pooling method")
    parser.add_argument("--reprocess", action="store_true", help="Force reprocessing even if files exist")
    parser.add_argument("--sentence_mask_type", type=str, default=None, help="Type of sentence masking used ('moral_word' or 'all')")
    args = parser.parse_args()

    print("Starting preprocessing for moral word prediction...")
    main(args)
    print("Preprocessing completed.")

