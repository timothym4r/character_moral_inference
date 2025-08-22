import json
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Load data

# TODO: Set the correct path to your data file
data_file_path = None

with open(data_file_path) as f:
    data_for_project_6 = json.load(f)

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

# Embedding function
def generate_embeddings(tokenizer, model, sentences, batch_size=32):
    all_embeddings = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
            all_embeddings.append(cls_embeddings.cpu())

        del inputs, outputs, cls_embeddings
        torch.cuda.empty_cache()

    return torch.cat(all_embeddings, dim=0).numpy()

# Generate embeddings for each model
for movie, characters in tqdm(data_for_project_6["sentences"].items(), desc="Processing movies"):
    bert_embeddings[movie] = {}
    roberta_embeddings[movie] = {}
    deberta_embeddings[movie] = {}
    electra_embeddings[movie] = {}

    for character, sentences in tqdm(characters.items(), desc=f"Processing characters in {movie}", leave=False):
        bert_embeddings[movie][character] = generate_embeddings(tokenizers["bert"], models["bert"], sentences)
        roberta_embeddings[movie][character] = generate_embeddings(tokenizers["roberta"], models["roberta"], sentences)
        deberta_embeddings[movie][character] = generate_embeddings(tokenizers["deberta"], models["deberta"], sentences)
        electra_embeddings[movie][character] = generate_embeddings(tokenizers["electra"], models["electra"], sentences)

import pickle
import os

output_dir = "/content/drive/MyDrive/Moral-inference/summer/moral-classification/with-embedding/"
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
