from torch.utils.data import Dataset
from transformers import BertTokenizer
import torch
import torch.nn as nn

from torch.utils.data import Dataset
from transformers import BertTokenizer
import torch
import torch.nn as nn

class MoralDataset(Dataset):
    def __init__(self, data, tokenizer=None, max_length=512, use_one_hot=False, char2id=None, embed_dim=768):
        self.data = data
        self.tokenizer = tokenizer or BertTokenizer.from_pretrained("bert-base-uncased")
        self.max_length = max_length
        self.use_one_hot = use_one_hot
        self.char2id = char2id
        self.embed_dim = embed_dim

        if self.use_one_hot and self.char2id is None:
            raise ValueError("char2id mapping must be provided when use_one_hot=True")

    def __len__(self):
        return len(self.data)

    def _to_embed_seq(self, maybe_list):
        """
        Converts a list (T x D) or tensor to a FloatTensor (T x D).
        Allows empty list -> (0 x D).
        """
        if maybe_list is None:
            return torch.zeros(0, self.embed_dim, dtype=torch.float32)

        if torch.is_tensor(maybe_list):
            x = maybe_list
        else:
            # JSON gives python lists; may be [] or list of lists
            if len(maybe_list) == 0:
                return torch.zeros(0, self.embed_dim, dtype=torch.float32)
            x = torch.tensor(maybe_list)

        # ensure float32 and shape (T, D)
        x = x.to(dtype=torch.float32)
        if x.dim() == 1:
            # edge case: a single vector saved as [D] (shouldn’t happen, but be robust)
            x = x.unsqueeze(0)
        return x

    def _to_embed_vec(self, maybe_list):
        """
        Converts a list (D) or tensor to FloatTensor (D).
        """
        if maybe_list is None:
            return torch.zeros(self.embed_dim, dtype=torch.float32)
        if torch.is_tensor(maybe_list):
            x = maybe_list
        else:
            x = torch.tensor(maybe_list)
        return x.to(dtype=torch.float32)

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

        mask_positions = (encoding["input_ids"] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        if len(mask_positions[1]) == 0:
            return {}  # you already filter these out upstream
        mask_index = mask_positions[1][0].item()

        result = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "target_id": torch.tensor(target_id, dtype=torch.long),
            "mask_index": torch.tensor(mask_index, dtype=torch.long),
            "movie": row["movie"],
            "character": row["character"],
        }

        if self.use_one_hot:
            key = f"{row['movie']}_{row['character']}"
            result["character_id"] = torch.tensor(self.char2id[key], dtype=torch.long)
        else:
            # NEW: two-stream history + means for attention pooling
            result["spoken_mean"] = self._to_embed_vec(row.get("spoken_mean"))
            result["action_mean"] = self._to_embed_vec(row.get("action_mean"))

            result["spoken_history_embeds"] = self._to_embed_seq(row.get("spoken_history_embeds"))
            result["action_history_embeds"] = self._to_embed_seq(row.get("action_history_embeds"))

            # Optional debug / ablation convenience:
            # If you kept these in preprocess, they can be useful but not required.
            # result["spoken_count"] = torch.tensor(row.get("spoken_count", 0), dtype=torch.long)
            # result["action_count"] = torch.tensor(row.get("action_count", 0), dtype=torch.long)

        return result

class Autoencoder(nn.Module):
    def __init__(self, input_dim=768, latent_dim=20, intermediate_dim = 256, dropout=0.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


class TwoStreamAttnPool(nn.Module):
    def __init__(self, hidden_dim=768):
        super().__init__()
        # query vectors (learned) for each stream
        self.q_spk = nn.Parameter(torch.empty(hidden_dim))
        self.q_act = nn.Parameter(torch.empty(hidden_dim))

        # mixture logits -> softmax -> weights (init 0 => 0.5/0.5)
        self.mix_logits = nn.Parameter(torch.zeros(2))

        nn.init.normal_(self.q_spk, mean=0.0, std=0.02)
        nn.init.normal_(self.q_act, mean=0.0, std=0.02)

    def attn_pool(self, X, mask, q):
        """
        X:    B x T x D
        mask: B x T  (1 valid, 0 pad)
        q:    D
        returns: pooled B x D
        """
        # scores: B x T
        scores = torch.einsum("btd,d->bt", X, q)

        # mask pads -> -inf
        scores = scores.masked_fill(mask <= 0, -1e9)
        attn = torch.softmax(scores, dim=1)  # B x T

        pooled = torch.einsum("bt,btd->bd", attn, X)
        return pooled

    def forward(self, spk_hist, spk_mask, act_hist, act_mask, spk_mean=None, act_mean=None):
        # If a stream is empty (all mask zeros), attention gives NaNs; fallback to mean if provided.
        spk_all_empty = (spk_mask.sum(dim=1) == 0)
        act_all_empty = (act_mask.sum(dim=1) == 0)

        c_spk = self.attn_pool(spk_hist, spk_mask, self.q_spk)
        c_act = self.attn_pool(act_hist, act_mask, self.q_act)

        if spk_mean is not None:
            c_spk = torch.where(spk_all_empty.unsqueeze(1), spk_mean, c_spk)
        if act_mean is not None:
            c_act = torch.where(act_all_empty.unsqueeze(1), act_mean, c_act)

        w = torch.softmax(self.mix_logits, dim=0)  # (2,)
        c = w[0] * c_spk + w[1] * c_act
        return c, c_spk, c_act, w

