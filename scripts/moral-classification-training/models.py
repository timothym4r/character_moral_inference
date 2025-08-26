import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel

import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim=768, latent_dim=20, dropout=0.1):
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


class MoralClassifier(nn.Module):
    def __init__(self, base_model, latent_dim=768, inject_operation = "sum"):
        super().__init__()
        self.bert = base_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 1)  # Binary classification (moral or not)
        self.operation = inject_operation

    def forward(self, input_ids, attention_mask, char_vec=None):
        """
        A forward function. This can be extended to support more operations.
        """

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token

        if char_vec is not None:
            if self.operation == "sum":
                cls_embedding = cls_embedding + char_vec  # Inject character info

        cls_embedding = self.dropout(cls_embedding)
        logits = self.classifier(cls_embedding)
        return logits


class MoralRelevanceDataset(Dataset):
    def __init__(self, data, tokenizer=None, max_length=256, use_one_hot=False, char2id=None):
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

        # No [MASK] — just classify the full sentence
        encoding = self.tokenizer(
            row["sentence"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )

        result = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(row["label"], dtype=torch.long)  # 0 or 1
        }

        # Inject character information (optional)
        if self.use_one_hot:
            row_character_name = row["movie"] + "_" + row["character"]
            result["character_id"] = torch.tensor(self.char2id[row_character_name], dtype=torch.long)
        else:
            result["avg_embedding"] = torch.tensor(row["avg_embedding"], dtype=torch.float)

        # Include the past sentences for regenerating embeddings as we update the classifier
        result["past_sentences"] = row["past_sentences"]
        result["movie"] = row["movie"]
        result["character"] = row['character']

        return result