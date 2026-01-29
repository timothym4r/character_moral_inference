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

        # 1) tokenize target word into wordpieces
        target_toks = self.tokenizer.tokenize(row["target_word"])
        if len(target_toks) == 0:
            raise ValueError(f"Empty tokenization for target_word={row['target_word']}")

        # 2) ensure masked_sentence has the right number of [MASK] tokens
        masked_sentence = row["masked_sentence"]

        # how many masks are currently in the sentence?
        tmp = self.tokenizer(masked_sentence, return_tensors="pt", truncation=True, max_length=self.max_length)
        cur_num_masks = (tmp["input_ids"][0] == self.tokenizer.mask_token_id).sum().item()

        # Common case in your pipeline: exactly 1 [MAXK] in text, but target has k wordpieces
        # Expand that single [MASK] into k masks.
        if cur_num_masks == 1 and len(target_toks) > 1:
            masked_sentence = masked_sentence.replace(
                self.tokenizer.mask_token,
                " ".join([self.tokenizer.mask_token] * len(target_toks)),
                1  # replace only the first occurrence
            )

        # If you already created multiple masks upstream, that's fine.
        # But now we must enforce: number of masks == number of target wordpieces
        encoding = self.tokenizer(
            masked_sentence,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )

        mask_positions = (encoding["input_ids"] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        mask_indices = mask_positions[1].tolist()  # list of positions in the sequence

        if len(mask_indices) == 0:
            return {}  # or raise

        if len(mask_indices) != len(target_toks):
            # This indicates your masked_sentence and target_word are not aligned.
            # Better to skip than train on wrong supervision.
            # You can print for debugging:
            # print("Mismatch:", row["target_word"], target_toks, "masks:", len(mask_indices), masked_sentence)
            return {}

        target_ids = self.tokenizer.convert_tokens_to_ids(target_toks)
        if any(tid == self.tokenizer.unk_token_id for tid in target_ids):
            # weird / bad targets; skip
            return {}

        result = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),

            # CHANGED: multi-token targets + multi mask indices
            "target_ids": torch.tensor(target_ids, dtype=torch.long),          # (k,)
            "mask_indices": torch.tensor(mask_indices, dtype=torch.long),      # (k,)

            "movie": row["movie"],
            "character": row["character"],
        }

        if self.use_one_hot:
            key = f"{row['movie']}_{row['character']}"
            result["character_id"] = torch.tensor(self.char2id[key], dtype=torch.long)
        else:
            result["spoken_mean"] = self._to_embed_vec(row.get("spoken_mean"))
            result["action_mean"] = self._to_embed_vec(row.get("action_mean"))
            result["spoken_history_embeds"] = self._to_embed_seq(row.get("spoken_history_embeds"))
            result["action_history_embeds"] = self._to_embed_seq(row.get("action_history_embeds"))

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

class TwoStreamMeanPool(nn.Module):
    def __init__(self, hidden_dim=768):
        super().__init__()
        # same fusion idea as your attn pooler
        self.mix_logits = nn.Parameter(torch.zeros(2))

    @staticmethod
    def masked_mean(X, mask, eps=1e-9):
        """
        X:    B x T x D
        mask: B x T  (1 valid, 0 pad)
        """
        m = mask.unsqueeze(-1)  # B x T x 1
        num = (X * m).sum(dim=1)              # B x D
        den = m.sum(dim=1).clamp(min=eps)     # B x 1
        return num / den

    def forward(self, spk_hist, spk_mask, act_hist, act_mask, spk_mean=None, act_mean=None):
        spk_all_empty = (spk_mask.sum(dim=1) == 0)
        act_all_empty = (act_mask.sum(dim=1) == 0)

        c_spk = self.masked_mean(spk_hist, spk_mask)
        c_act = self.masked_mean(act_hist, act_mask)

        # fallback if empty (optional but consistent with your current design)
        if spk_mean is not None:
            c_spk = torch.where(spk_all_empty.unsqueeze(1), spk_mean, c_spk)
        if act_mean is not None:
            c_act = torch.where(act_all_empty.unsqueeze(1), act_mean, c_act)

        w = torch.softmax(self.mix_logits, dim=0)  # (2,)
        c = w[0] * c_spk + w[1] * c_act
        return c, c_spk, c_act, w

class TwoStreamMovingAvgPool(nn.Module):
    def __init__(self, hidden_dim=768, decay=0.9, learn_decay=False):
        super().__init__()
        self.mix_logits = nn.Parameter(torch.zeros(2))

        # Optionally learn decay (in (0,1)) using sigmoid parameterization
        if learn_decay:
            # initialize so sigmoid(param) ~= decay
            init = torch.log(torch.tensor(decay) / (1 - torch.tensor(decay)))
            self.decay_logit = nn.Parameter(init.clone().float())
        else:
            self.register_buffer("decay_const", torch.tensor(float(decay)))
            self.decay_logit = None

    def _decay(self):
        if self.decay_logit is None:
            return self.decay_const
        return torch.sigmoid(self.decay_logit)

    def ema_pool(self, X, mask):
        """
        X:    B x T x D
        mask: B x T  (1 valid, 0 pad)
        Returns B x D
        """
        B, T, D = X.shape
        decay = self._decay().to(X.device)

        # We'll do an EMA scan:
        # h_t = decay*h_{t-1} + (1-decay)*x_t, but only when mask=1.
        h = torch.zeros(B, D, device=X.device, dtype=X.dtype)
        has_any = torch.zeros(B, 1, device=X.device, dtype=X.dtype)  # track if any valid has appeared

        one_minus = (1.0 - decay)

        for t in range(T):
            mt = mask[:, t].unsqueeze(1)  # B x 1
            xt = X[:, t, :]               # B x D

            # update only where mt==1
            h_new = decay * h + one_minus * xt
            h = mt * h_new + (1.0 - mt) * h

            has_any = torch.clamp(has_any + mt, max=1.0)

        return h, has_any.squeeze(1)  # (B x D), (B,) indicates non-empty

    def forward(self, spk_hist, spk_mask, act_hist, act_mask, spk_mean=None, act_mean=None):
        c_spk, spk_nonempty = self.ema_pool(spk_hist, spk_mask)
        c_act, act_nonempty = self.ema_pool(act_hist, act_mask)

        # fallback if completely empty
        if spk_mean is not None:
            c_spk = torch.where((spk_nonempty == 0).unsqueeze(1), spk_mean, c_spk)
        if act_mean is not None:
            c_act = torch.where((act_nonempty == 0).unsqueeze(1), act_mean, c_act)

        w = torch.softmax(self.mix_logits, dim=0)
        c = w[0] * c_spk + w[1] * c_act
        return c, c_spk, c_act, w
