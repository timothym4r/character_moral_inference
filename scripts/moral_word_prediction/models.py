from torch.utils.data import Dataset
from transformers import BertTokenizer
import torch
import torch.nn as nn

class MoralDataset(Dataset):
    def __init__(self, data, tokenizer=None, max_length=512, use_one_hot=False, char2id=None):
        self.data = data
        self.tokenizer = tokenizer or BertTokenizer.from_pretrained("bert-base-uncased")
        self.max_length = max_length
        self.use_one_hot = use_one_hot
        self.char2id = char2id

        if self.use_one_hot and self.char2id is None:
            raise ValueError("char2id mapping must be provided when use_one_hot=True")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        encoding = self.tokenizer(
            row["masked_sentence"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )

        target_id = self.tokenizer.convert_tokens_to_ids(row["target_word"])

        # With this safe logic:
        mask_positions = (encoding["input_ids"] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        if len(mask_positions[1]) == 0:
            return {}
            # raise ValueError(f"[MASK] token not found in input: {row['masked_sentence']}")
        mask_index = mask_positions[1][0].item()

        result = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "target_id": torch.tensor(target_id, dtype=torch.long),
            "mask_index": torch.tensor(mask_index, dtype=torch.long),
            "movie": row["movie"],
            "character": row["character"],
            "past_sentences": row["past_sentences"]
        }

        if self.use_one_hot:
            row_character_name = row["movie"] + "_" + row["character"]
            result["character_id"] = torch.tensor(self.char2id[row_character_name], dtype=torch.long)
        else:
            result["avg_embedding"] = torch.tensor(row["avg_embedding"], dtype=torch.float)

        return result


class Autoencoder(nn.Module):
    def __init__(self, input_dim=768, latent_dim=20, dropout=0.0):
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

