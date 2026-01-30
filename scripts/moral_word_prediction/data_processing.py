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

def safe_mean(x: torch.Tensor, dim=0, out_dim=None):
    """Mean with empty-safe fallback."""
    if x is None or x.numel() == 0:
        assert out_dim is not None
        return torch.zeros(out_dim, dtype=torch.float32)
    return x.mean(dim=dim)
 
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

def data_preprocess(
    model_name,
    source_data_path,
    output_dir,
    threshold=20,
    pooling_method="mean",
    reprocess=False,
    sentence_mask_type=None,
    # NEW FLAGS
    add_type_tokens=True,
    store_history_embeddings=True,
    max_history_per_type=None,   # e.g. 200 to cap record size; None = no cap
    save_fp16=True               # store history embeds as float16 to reduce JSON size
):
    os.makedirs(output_dir, exist_ok=True)

    if sentence_mask_type is not None:
        train_path = os.path.join(output_dir, f"train_data_{pooling_method}_{threshold}_{sentence_mask_type}.json")
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

    if add_type_tokens:
        ensure_special_tokens(tokenizer, model, ["[SPK]", "[ACT]"])

    print("Loading source data...")
    with open(source_data_path, "r") as f:
        moral_data = json.load(f)

    moral_dialogue = moral_data["moral_dialogue"]
    moral_dialogue_masked = moral_data["moral_dialogue_masked"]
    ground_truths = moral_data["ground_truths"]
    sentence_type = moral_data["sentence_type"]  # "spoken" or "action"

    if sentence_mask_type is not None:
        mask_prediction_index = moral_data["moral_label"]

    all_records = []

    for movie, characters in tqdm(moral_dialogue.items(), desc="Processing characters"):
        for character, original_sentences in characters.items():
            num_sentences = len(original_sentences)

            if num_sentences < threshold:
                continue

            try:
                # Convert sentence_type list -> np array for correct boolean masking
                stypes = np.array(sentence_type[movie][character], dtype=object)
                assert len(stypes) == num_sentences

                # Option A: prefix sentences by type BEFORE embedding
                if add_type_tokens:
                    typed_sentences = [prefix_by_type(s, t) for s, t in zip(original_sentences, stypes)]
                else:
                    typed_sentences = original_sentences

                embeddings = get_sentence_embeddings(
                    typed_sentences, model, tokenizer, model.device, pooling_method=pooling_method
                )  # [num_sentences, hidden]

                masked_sentences = moral_dialogue_masked[movie][character]
                moral_words = ground_truths[movie][character]

                for idx in range(threshold, num_sentences):
                    if sentence_mask_type is not None and mask_prediction_index[movie][character][idx] == "No":
                        continue

                    # History up to idx (exclusive)
                    hist_mask = np.arange(idx)
                    hist_types = stypes[hist_mask]

                    spoken_mask = (hist_types == "spoken")
                    action_mask = (hist_types == "action")

                    spoken_embeds = embeddings[:idx][torch.from_numpy(spoken_mask)]
                    action_embeds = embeddings[:idx][torch.from_numpy(action_mask)]

                    # Optional cap to reduce JSON size
                    if max_history_per_type is not None:
                        if spoken_embeds.size(0) > max_history_per_type:
                            spoken_embeds = spoken_embeds[-max_history_per_type:]
                        if action_embeds.size(0) > max_history_per_type:
                            action_embeds = action_embeds[-max_history_per_type:]

                    hidden_dim = embeddings.size(1)
                    spoken_mean = safe_mean(spoken_embeds, dim=0, out_dim=hidden_dim)
                    action_mean = safe_mean(action_embeds, dim=0, out_dim=hidden_dim)

                    # Prefix the *current* masked sentence with its type token too
                    cur_type = stypes[idx]
                    cur_masked_sentence = masked_sentences[idx]
                    if add_type_tokens:
                        cur_masked_sentence = prefix_by_type(cur_masked_sentence, cur_type)

                    record = {
                        "movie": movie,
                        "character": character,
                        "sentence_type": str(cur_type),  # "spoken" or "action"
                        "masked_sentence": cur_masked_sentence,
                        "target_word": moral_words[idx],
                        "history_len": int(idx),

                        # Two-stream pooled baselines (useful ablations / fallback)
                        "spoken_mean": spoken_mean.tolist(),
                        "action_mean": action_mean.tolist(),
                        "spoken_count": int(spoken_embeds.size(0)),
                        "action_count": int(action_embeds.size(0)),
                    }

                    # Proper attention-pooling inputs: store sequences per type
                    if store_history_embeddings:
                        # store as fp16 list to reduce file size (still JSON)
                        if save_fp16:
                            se = spoken_embeds.numpy().astype(np.float16).tolist()
                            ae = action_embeds.numpy().astype(np.float16).tolist()
                        else:
                            se = spoken_embeds.numpy().astype(np.float32).tolist()
                            ae = action_embeds.numpy().astype(np.float32).tolist()

                        record["spoken_history_embeds"] = se
                        record["action_history_embeds"] = ae

                    all_records.append(record)

            except RuntimeError as e:
                print(f"Skipping {character} from {movie} due to memory error: {e}")
                torch.cuda.empty_cache()
                gc.collect()

    # ---- Movie-level split (no movie appears in both) ----
    movies = sorted({r["movie"] for r in all_records})
    random.shuffle(movies)

    split_idx = int(0.7 * len(movies))
    train_movies = set(movies[:split_idx])
    test_movies  = set(movies[split_idx:])

    train_data = [r for r in all_records if r["movie"] in train_movies]
    test_data  = [r for r in all_records if r["movie"] in test_movies]

    print(f"Movie split: {len(train_movies)} train movies / {len(test_movies)} test movies")
    print(f"Record split: {len(train_data)} train records / {len(test_data)} test records")

    # (Optional) sanity check
    assert set(r["movie"] for r in train_data).isdisjoint(set(r["movie"] for r in test_data))

    with open(train_path, "w") as f:
        json.dump(train_data, f)
    with open(test_path, "w") as f:
        json.dump(test_data, f)

    print(f"Saved train/test data at {output_dir} ({len(train_data)} train / {len(test_data)} test).")

def main(args):

    # NOTE: We can hardcode some flags here for moral word prediction

    data_preprocess(
        model_name=args.model_name,
        source_data_path=args.source_data_path,
        output_dir=args.output_dir,
        threshold=args.threshold,
        pooling_method=args.pooling_method,
        reprocess=args.reprocess,
        sentence_mask_type=args.sentence_mask_type,
        add_type_tokens=args.add_type_tokens,
        store_history_embeddings=args.store_history_embeddings,
        max_history_per_type=args.max_history_per_type,
        save_fp16=True,
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

