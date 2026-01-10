# Import Libraries and Tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import random
from collections import defaultdict
from tqdm import tqdm
import re
import os
import csv

from torch.optim import AdamW
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# Language Models
from transformers import get_scheduler

from transformers import AutoModelForMaskedLM, AutoTokenizer

# Classification Models
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Import from local modules
from models import MoralDataset
from models import Autoencoder
import argparse

from utils import normalize_mask_token

random.seed(42)

def filter_maskless_entries(data, tokenizer, max_length=512):
    mask_token_id = tokenizer.mask_token_id
    cleaned = []
    for row in data:
        encoding = tokenizer(
            row["masked_sentence"],
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        if mask_token_id in encoding["input_ids"][0]:
            cleaned.append(row)
    return cleaned


def precompute_sentence_embeddings(dataset, tokenizer, encoder, device, batch_size=64, pooling_method="mean", mask_index = None):
    """
    Precompute embeddings for each unique (movie, character).
    We only encode the *full* list of past sentences once, then reuse it for all rows.
    Returns dict: (movie, character) -> tensor [num_sentences, embed_dim]
    """
    encoder.eval()
    char2sentences = {}

    # Collect the longest past_sentences per (movie, character)
    for row in dataset:
        key = (row["movie"], row["character"])
        if key not in char2sentences or len(row["past_sentences"]) > len(char2sentences[key]):
            char2sentences[key] = row["past_sentences"]

    cache = {}
    with torch.no_grad():
        for key, sentences in tqdm(char2sentences.items(), desc="Precomputing char embeddings"):
            all_embeddings = []
            for j in range(0, len(sentences), batch_size):
                batch = sentences[j:j+batch_size]
                encoded = tokenizer(batch, return_tensors="pt", padding=True,
                                    truncation=True, max_length=256).to(device)
                outputs = encoder(**encoded)
                last_hidden = outputs.last_hidden_state
                if pooling_method == "mean":
                    attn_mask = encoded["attention_mask"].unsqueeze(-1)
                    pooled = (last_hidden * attn_mask).sum(1) / attn_mask.sum(1).clamp(min=1e-9)
                else:  # 'cls'
                    pooled = last_hidden[:, 0, :]
                all_embeddings.append(pooled.cpu())
            cache[key] = torch.cat(all_embeddings, dim=0)  # [num_sentences, hidden_dim]

    return cache


def update_char_vec_embeddings(dataset, sentence_cache):
    """
    Use precomputed embeddings per (movie, character).
    For each row, take the mean of the slice up to len(past_sentences).
    """
    for row in tqdm(dataset, desc="Updating char embeddings"):
        key = (row["movie"], row["character"])
        past_len = len(row["past_sentences"])
        if past_len == 0:
            row["avg_embedding"] = torch.zeros(768)
            continue
        full_embeds = sentence_cache[key]  # [num_sentences, hidden_dim]
        row["avg_embedding"] = full_embeds[:past_len].mean(0)


def init_csv(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)  # Create parent dirs if needed
    header = [
        "model", "epoch",
        # training losses
        "train_total_loss", "train_recon_loss", "train_ce_loss", "train_kl_loss", "train_ort_loss",
        # eval metrics
        "eval_cross_entropy", "eval_perplexity",
        "eval_accuracy@1", "eval_accuracy@5", "eval_accuracy@10"
    ]
    new_file = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(header)


def log_metrics_to_csv(log_path, model_name, epoch, train_losses, eval_metrics):
    row = [
        model_name, epoch,
        train_losses["total"], train_losses["recon"], train_losses["ce"], train_losses["kl"], train_losses["ort"],
        eval_metrics["cross_entropy"], eval_metrics["perplexity"],
        eval_metrics["accuracy@1"], eval_metrics["accuracy@5"], eval_metrics["accuracy@10"]
    ]
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

def compute_orthogonality_loss(z, strategy="off_diag"):
    """
    Compute orthogonality loss on the encoded matrix z.

    Args:
        z (torch.Tensor): shape (batch_size, latent_dim)
        strategy (str): currently supports "off_diag" only

    Returns:
        ortho_loss (torch.Tensor): scalar tensor
    """
    E = F.normalize(z, p=2, dim=1)  # Normalize each row
    EtE = torch.matmul(E.T, E) / E.size(0)

    if strategy == "off_diag":
        # Subtract diagonal (identity), penalize only off-diagonal
        off_diag = EtE - torch.diag_embed(torch.diagonal(EtE))
        ortho_loss = torch.norm(off_diag, p='fro')
        return ortho_loss
    else:
        raise ValueError(f"Unsupported orthogonality loss strategy: {strategy}")

def custom_collate_fn(batch):
    tensor_keys = ["input_ids", "attention_mask", "target_id", "mask_index"]
    optional_keys = ["character_id", "avg_embedding"]

    out = {key: [] for key in tensor_keys + optional_keys}
    out["movie"], out["character"], out["past_sentences"] = [], [], []

    for sample in batch:
        # guaranteed tensor keys
        for key in tensor_keys:
            out[key].append(sample[key])

        # optional keys
        for key in optional_keys:
            if key in sample:   # only append if exists
                out[key].append(sample[key])

        # keep as lists
        out["movie"].append(sample["movie"])
        out["character"].append(sample["character"])
        out["past_sentences"].append(sample["past_sentences"])

    # Stack guaranteed tensor keys
    for key in tensor_keys:
        out[key] = torch.stack(out[key])

    # Stack optional only if non-empty
    for key in optional_keys:
        if len(out[key]) > 0:
            out[key] = torch.stack(out[key])

    return out

def get_lm_head(model):
    if hasattr(model, "cls"):
        return model.cls          # BERT
    if hasattr(model, "lm_head"):
        return model.lm_head      # RoBERTa
    raise ValueError("Unsupported model: no cls or lm_head found.")

def freeze_all_params(model):
    for p in model.parameters():
        p.requires_grad = False

def unfreeze_last_n_transformer_layers(model, n_last: int):
    """
    Unfreeze last n_last transformer blocks in the base model (bert/roberta/...).
    Works for BERT and RoBERTa MLM models from HF.
    """
    if n_last <= 0:
        return

    prefix = model.base_model_prefix  # "bert" or "roberta"
    n_layers = model.config.num_hidden_layers
    start = max(0, n_layers - n_last)

    for name, p in model.named_parameters():
        # examples:
        # bert.encoder.layer.11.attention...
        # roberta.encoder.layer.11.attention...
        if name.startswith(f"{prefix}.encoder.layer."):
            # pick layer id from the name
            # name like "{prefix}.encoder.layer.{i}...."
            parts = name.split(".")
            # parts: [prefix, "encoder", "layer", "{i}", ...]
            if len(parts) > 3 and parts[3].isdigit():
                layer_id = int(parts[3])
                if layer_id >= start:
                    p.requires_grad = True

def train_mlm_model(
    train_dataset, val_dataset,
    use_vae=False, use_one_hot=False, char2id=None,
    latent_dim=20, alpha=1.0, beta=0.01, num_epochs=5, batch_size=32,
    lr_ae=1e-3, lr_bert=2e-5,
    dropout_rate=0.1, clip_grad_norm=5.0, weight_decay=1e-5, pooling_method = "mean",
    scheduler_type="cosine", early_stopping_patience=3,
    train_n_last_layers=0, log_path=None, inject_embedding=True, model_name =  "bert-base-uncased", eval_only=False, 
    moving_avg = False, moving_avg_window = -1 # moving_avg_window = -1 means we don't use windowed moving average
):
    # Load pretrained model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_lm = AutoModelForMaskedLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

    if eval_only:
        print(f"Running evaluation-only mode for {model_name} (no fine-tuning)...")
        eval_metrics = evaluate_mlm(
            model_H=None,
            dataset=val_dataset,
            tokenizer=tokenizer,
            bert_lm=bert_lm,
            use_one_hot=False,
            character_embedding=None,
            inject_embedding=False
        )
        print(f"[Baseline Eval] CE={eval_metrics['cross_entropy']:.4f}, "
              f"PPL={eval_metrics['perplexity']:.4f}, "
              f"Acc@1={eval_metrics['accuracy@1']:.4f}, "
              f"Acc@5={eval_metrics['accuracy@5']:.4f}, "
              f"Acc@10={eval_metrics['accuracy@10']:.4f}")
        return None, None, bert_lm  # nothing trained
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_lm.to(device)

    if hasattr(bert_lm, "bert"):
        encoder = bert_lm.bert
    elif hasattr(bert_lm, "roberta"):
        encoder = bert_lm.roberta
    else:
        raise ValueError("Unsupported architecture")

    # Freeze all params first
    freeze_all_params(bert_lm)
    # for name, param in bert_lm.named_parameters():
    #     param.requires_grad = False

    # Unfreeze the last n transformer layers
    unfreeze_last_n_transformer_layers(bert_lm, train_n_last_layers)
    # if train_n_last_layers > 0:
    #     for name, param in bert_lm.named_parameters():
    #         if any(f"encoder.layer.{i}" in name for i in range(12-train_n_last_layers, 12)):
    #             param.requires_grad = True

    # Allow cls/lm_head layer being trained
    lm_head = get_lm_head(bert_lm)
    for _, p in lm_head.named_parameters():
        p.requires_grad = True

    ### H-module
    # NOTE: VAE might be added later
    if use_vae:
        raise NotImplementedError("VAE mode is not yet supported.")
        # return None, None, None
    model_H = Autoencoder(768, latent_dim, dropout=dropout_rate)
    model_H.to(device)

    # Character embeddings if one-hot
    if use_one_hot:
        if char2id is None:
            raise ValueError("char2id required when use_one_hot=True")
        character_embedding = nn.Embedding(len(char2id), 768).to(device)
        embedding_params = list(character_embedding.parameters())
    else:
        character_embedding = None
        embedding_params = []

    # Optimizer: separate LRs
    optim_groups = [
        {"params": [p for n, p in bert_lm.named_parameters() if p.requires_grad], "lr": lr_bert},
    ]

    if inject_embedding:
        optim_groups.insert(0, {"params": model_H.parameters(), "lr": lr_ae})
        if embedding_params:
            optim_groups.insert(1, {"params": embedding_params, "lr": lr_ae})

    # Optimizer
    optimizer = AdamW(optim_groups, weight_decay=weight_decay)

    # Dataloader
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

    # Scheduler
    if scheduler_type == "cosine":
        num_training_steps = len(loader) * num_epochs
        num_warmup_steps = int(0.1 * num_training_steps)
        scheduler = get_scheduler("cosine", optimizer, num_warmup_steps, num_training_steps)
    elif scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    elif scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    else:
        scheduler = None

    # Tracking best model
    best_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(num_epochs):
        if inject_embedding:
            model_H.train()
        if character_embedding:
            character_embedding.train()

        total_loss, recon_total, ce_total, kl_total, ort_total = 0, 0, 0, 0, 0

        # ---- Training loop ----
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_id = batch["target_id"].to(device)
            mask_index = batch["mask_index"]

            if inject_embedding:
                # character vec
                if use_one_hot:
                    char_id = batch["character_id"].to(device)
                    char_vec = character_embedding(char_id)
                else:
                    char_vec = batch["avg_embedding"].to(device)

                # forward H
                if use_vae:
                    recon_vec, mu, logvar, z = model_H(char_vec)
                    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / char_vec.size(0)
                else:
                    recon_vec, z = model_H(char_vec)
                    kl_div = 0

                requires_grad_bert = any(p.requires_grad for p in encoder.parameters())
                if requires_grad_bert:
                    hidden = encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
                else:
                    with torch.no_grad():
                        hidden = encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

                # inject
                for i, mi in enumerate(mask_index):
                    hidden[i, mi, :] += recon_vec[i]

                lm_head = get_lm_head(bert_lm)
                logits = lm_head(hidden)

                mask_logits = torch.stack([logits[i, mi] for i, mi in enumerate(mask_index)])

                # losses
                ce_loss = nn.CrossEntropyLoss()(mask_logits, target_id)
                recon_loss = nn.MSELoss()(recon_vec, char_vec)
                if batch_idx % 4 == 0:  # optional, same as classifier
                    ortho_loss = compute_orthogonality_loss(z, strategy="off_diag")
                else:
                    ortho_loss = torch.tensor(0.0, device=z.device)

                loss = recon_loss + alpha * ce_loss + (alpha * kl_div if use_vae else 0) + beta * ortho_loss

            else:
                # no injection: vanilla MLM
                logits = bert_lm(input_ids=input_ids, attention_mask=attention_mask).logits
                mask_logits = torch.stack([logits[i, mi] for i, mi in enumerate(mask_index)])
                ce_loss = nn.CrossEntropyLoss()(mask_logits, target_id)

                # define placeholders so accumulators don’t crash
                recon_loss = torch.tensor(0.0, device=logits.device)
                kl_div     = torch.tensor(0.0, device=logits.device)
                ortho_loss = torch.tensor(0.0, device=logits.device)

                loss = alpha * ce_loss

            # backward
            optimizer.zero_grad()
            loss.backward()
            if clip_grad_norm:
                nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]], clip_grad_norm)
            optimizer.step()
            if scheduler_type == "cosine": scheduler.step()

            # accumulate
            total_loss += loss.item()
            recon_total += recon_loss.item()
            ce_total += ce_loss.item()
            kl_total += kl_div.item() if use_vae else 0
            ort_total += (beta * ortho_loss).item()

        # ---- Evaluation after epoch ----
        eval_metrics = evaluate_mlm(model_H, val_dataset, tokenizer, bert_lm, use_one_hot, character_embedding, inject_embedding = inject_embedding)
        val_loss = eval_metrics["cross_entropy"]

        print(f"Epoch {epoch+1}: TrainTotal={total_loss:.4f}, Recon={recon_total:.4f}, CE={ce_total:.4f}, Ortho={ort_total:.4f}, KL={kl_total:.4f}")
        print(f"  Eval: CE={val_loss:.4f}, PPL={eval_metrics['perplexity']:.4f}, Acc@1={eval_metrics['accuracy@1']:.4f}, Acc@5={eval_metrics['accuracy@5']:.4f}, Acc@10={eval_metrics['accuracy@10']:.4f}")

        # Scheduler step for step/plateau schedulers
        if scheduler_type == "step":
            scheduler.step()
        elif scheduler_type == "plateau":
            scheduler.step(val_loss)

        # Recompute sentence embeddings if BERT is being fine-tuned
        if train_n_last_layers > 0:
            print(f"[Epoch {epoch+1}] Recomputing sentence embeddings for character vectors...")
            train_cache = precompute_sentence_embeddings(
                train_dataset, tokenizer, encoder, device, batch_size=64, pooling_method=pooling_method
            )
            val_cache = precompute_sentence_embeddings(
                val_dataset, tokenizer, encoder, device, batch_size=64, pooling_method=pooling_method
            )
            update_char_vec_embeddings(train_dataset, train_cache)
            update_char_vec_embeddings(val_dataset, val_cache)

        # ---- Logging ----
        if log_path:
            log_metrics_to_csv(
                log_path,
                model_name="bert-base-uncased-mlm",
                epoch=epoch + 1,
                train_losses={
                    "total": total_loss,
                    "recon": recon_total,
                    "ce": ce_total,
                    "kl": kl_total,
                    "ort": ort_total
                },
                eval_metrics=eval_metrics
            )

        # ---- Early stopping ----
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_state = {
                "model_H": model_H.state_dict(),
                "character_embedding": character_embedding.state_dict() if character_embedding else None,
                "bert_lm": bert_lm.state_dict()
            }
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}, best val CE={best_loss:.4f}")
                break

    # ---- Restore best model ----
    if best_state:
        if inject_embedding:
            model_H.load_state_dict(best_state["model_H"])
        else:
            model_H = None
        if character_embedding and best_state["character_embedding"]:
            character_embedding.load_state_dict(best_state["character_embedding"])
        if "bert_lm" in best_state and best_state["bert_lm"]:
            bert_lm.load_state_dict(best_state["bert_lm"])

    return model_H, character_embedding, bert_lm

def evaluate_mlm(model_H, dataset, tokenizer, bert_lm, use_one_hot=False, character_embedding=None, batch_size=16, inject_embedding = True):
    if inject_embedding and model_H is not None:
        model_H.eval()
    if character_embedding:
        character_embedding.eval()
    bert_lm.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device moves
    if inject_embedding and model_H is not None:
        model_H.to(device)
    bert_lm.to(device)
    if character_embedding:
        character_embedding.to(device)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    topk_hits = {1: 0, 5: 0, 10: 0}
    total, total_ce_loss = 0, 0
    
    # Selecting encoder layer
    if hasattr(bert_lm, "bert"):
        encoder = bert_lm.bert
    elif hasattr(bert_lm, "roberta"):
        encoder = bert_lm.roberta

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_id = batch["target_id"].to(device)
            mask_index = batch["mask_index"]

            if use_one_hot:
                char_id = batch["character_id"].to(device)
                char_vec = character_embedding(char_id)
            else:
                char_vec = batch["avg_embedding"].to(device)

            if inject_embedding:
                recon_vec, z = model_H(char_vec)
                hidden = encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
                for i, mi in enumerate(mask_index):
                    hidden[i, mi, :] += recon_vec[i]
                lm_head = get_lm_head(bert_lm)
                logits = lm_head(hidden)
                # logits = bert_lm.cls(hidden)
            else:
                logits = bert_lm(input_ids=input_ids, attention_mask=attention_mask).logits

            mask_logits = torch.stack([logits[i, mi] for i, mi in enumerate(mask_index)])

            probs = F.softmax(mask_logits, dim=-1)
            total += input_ids.size(0)

            for k in topk_hits.keys():
                topk_preds = torch.topk(probs, k=k, dim=-1).indices
                topk_hits[k] += (topk_preds == target_id.unsqueeze(1)).sum().item()

            ce_loss = F.cross_entropy(mask_logits, target_id, reduction="sum")
            total_ce_loss += ce_loss.item()

    results = {"cross_entropy": total_ce_loss / total if total > 0 else 0}
    results["perplexity"] = torch.exp(torch.tensor(results["cross_entropy"])).item()
    for k, correct in topk_hits.items():
        results[f"accuracy@{k}"] = correct / total if total > 0 else 0

    return results

def main(args):
    """
    Main function to train a Masked Language Model (MLM) with optional character embeddings.
    This script provides a command-line interface for training an MLM model with various configurations,
    including the use of Variational Autoencoders (VAE) or one-hot encoding for character embeddings. It
    supports fine-tuning BERT-based models, gradient clipping, learning rate scheduling, and early stopping.
    Command-line Arguments:
    -----------------------
    --input_dir : str (required)
        Directory containing the input JSON files (e.g., train_data.json, test_data.json).
    --output_dir : str (default: "models/mlm_model")
        Directory to save the trained models and logs.
    --use_vae : flag
        Use Variational Autoencoder for character embeddings.
    --use_one_hot : flag
        Use one-hot encoding for character embeddings.
    --latent_dim : int (default: 20)
        Latent dimension for the autoencoder.
    --alpha : float (default: 1.0)
        Weight for the cross-entropy loss.
    --beta : float (default: 0.01)
        Weight for the orthogonality loss.
    --num_epochs : int (default: 5)
        Number of training epochs.
    --batch_size : int (default: 32)
        Batch size for training.
    --lr_ae : float (default: 1e-3)
        Learning rate for the autoencoder.
    --lr_bert : float (default: 2e-5)
        Learning rate for the BERT model.
    --dropout_rate : float (default: 0.1)
        Dropout rate for the autoencoder.
    --clip_grad_norm : float (default: 5.0)
        Gradient clipping norm value.
    --weight_decay : float (default: 1e-5)
        Weight decay for the optimizer.
    --scheduler_type : str (default: "cosine")
        Type of learning rate scheduler. Choices: ["cosine", "step", "plateau"].
    --early_stopping_patience : int (default: 3)
        Patience for early stopping.
    --train_n_last_layers : int (default: 0)
        Number of last layers of BERT to fine-tune.
    --inject_embedding : flag
        Inject character embeddings into the model.
    --pooling_method : str (default: "mean")
        Pooling method for sentence embeddings. Choices: ["mean", "cls"].
    --retrain : flag
        Retrain the model even if saved models exist.
    Functionality:
    --------------
    1. Parses command-line arguments.
    2. Checks if the model already exists; skips training unless --retrain is specified.
    3. Loads training and validation data from JSON files.
    4. Initializes datasets and trains the MLM model with the specified configurations.
    5. Saves the trained model, character embeddings, and logs to the output directory.
    Notes:
    ------
    - The script assumes the input directory contains `train_data.json` and `test_data.json`.
    - Logs are saved in CSV format in the `logs` subdirectory of the output directory.
    - If the model already exists and --retrain is not specified, training is skipped.
    
    Example Usage:
    --------------
    python moral_word_prediction_training.py --input_dir data/ --output_dir models/ --use_vae --num_epochs 10
    """

    model_H_path = os.path.join(args.output_dir, "model_H.pth")
    bert_lm_path = os.path.join(args.output_dir, "bert_lm.pth")
    log_path = os.path.join(args.output_dir, "logs", "mlm_training_log.csv")
    # We document all the arguments used for training in a text file
    arg_log_txt_file = os.path.join(args.output_dir, f"{args.model_name}_mlm_args.txt")
    with open(arg_log_txt_file, "w") as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    if args.retrain or not (os.path.exists(model_H_path) and os.path.exists(bert_lm_path)):
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "training_args.txt"), "w") as f:
            for arg, value in vars(args).items():
                f.write(f"{arg}: {value}\n")

        if args.sentence_mask_type is not None:
            with open(os.path.join(args.input_dir, f"train_data_{args.pooling_method}_{args.threshold}_{args.sentence_mask_type}.json"), "r") as f:
                train_data = json.load(f)

            with open(os.path.join(args.input_dir, f"test_data_{args.pooling_method}_{args.threshold}_{args.sentence_mask_type}.json"), "r") as f:
                test_data = json.load(f)
        else:
            with open(os.path.join(args.input_dir, f"train_data_{args.pooling_method}_{args.threshold}.json"), "r") as f:
                train_data = json.load(f)

            with open(os.path.join(args.input_dir, f"test_data_{args.pooling_method}_{args.threshold}.json"), "r") as f:
                test_data = json.load(f)

        # Normalize [MASK] tokens
        train_data = normalize_mask_token(train_data, tokenizer)
        test_data = normalize_mask_token(test_data, tokenizer)

        # We filter out entries that do not have any [MASK]/<mask> tokens after tokenization
        train_data = filter_maskless_entries(train_data, tokenizer)
        test_data = filter_maskless_entries(test_data, tokenizer)

        init_csv(log_path)

        char2id = None
        if args.use_one_hot:
            # Build mapping of unique characters
            all_characters = set()
            for split in [train_data, test_data]:
                for row in split:
                    key = f"{row['movie']}_{row['character']}"
                    all_characters.add(key)

            char2id = {char: idx for idx, char in enumerate(sorted(all_characters))}
            id2char = {v: k for k, v in char2id.items()}
            print(f"Loaded {len(char2id)} unique characters.")
            with open(os.path.join(args.output_dir, "char2id.json"), "w") as f:
                json.dump(char2id, f, indent=2)


        train_dataset = MoralDataset(train_data, tokenizer=tokenizer, use_one_hot=args.use_one_hot, char2id=char2id)
        val_dataset = MoralDataset(test_data, tokenizer=tokenizer, use_one_hot=args.use_one_hot, char2id=char2id)

        model_H, character_embedding, bert_lm = train_mlm_model(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            use_vae=args.use_vae,
            use_one_hot=args.use_one_hot,
            char2id=char2id,
            latent_dim=args.latent_dim,
            alpha=args.alpha,
            beta=args.beta,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            lr_ae=args.lr_ae,
            lr_bert=args.lr_bert,
            dropout_rate=args.dropout_rate,
            clip_grad_norm=args.clip_grad_norm,
            weight_decay=args.weight_decay,
            pooling_method=args.pooling_method,
            scheduler_type=args.scheduler_type,
            early_stopping_patience=args.early_stopping_patience,
            train_n_last_layers=args.train_n_last_layers,
            log_path=log_path,
            inject_embedding=args.inject_embedding,
            model_name=args.model_name,
            eval_only=args.eval_only,
            # moving_avg = args.moving_avg,
            # moving_avg_window = args.moving_avg_window
        )

        if model_H:
            torch.save(model_H.state_dict(), model_H_path)
        if bert_lm:
            torch.save(bert_lm.state_dict(), bert_lm_path)
        if character_embedding:
            character_embedding_path = os.path.join(args.output_dir, "character_embedding.pth")
            torch.save(character_embedding.state_dict(), character_embedding_path)
    else:
        print("Model already exists. Use --retrain to overwrite.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLM Model with Character Embeddings")

    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing the input JSON files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the trained models and logs")
    parser.add_argument("--use_vae", action="store_true", help="Use Variational Autoencoder for character embeddings")
    parser.add_argument("--use_one_hot", action="store_true", help="Use one-hot encoding for character embeddings")
    parser.add_argument("--latent_dim", type=int, default=20, help="Latent dimension for the autoencoder")
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight for cross-entropy loss")
    parser.add_argument("--beta", type=float, default=0.01, help="Weight for orthogonality loss")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr_ae", type=float, default=1e-3, help="Learning rate for the autoencoder")
    parser.add_argument("--lr_bert", type=float, default=2e-5, help="Learning rate for the BERT model")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate for the autoencoder")
    parser.add_argument("--clip_grad_norm", type=float, default=5.0, help="Gradient clipping norm value")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for the optimizer")
    parser.add_argument("--scheduler_type", type=str, default="cosine", choices=["cosine", "step", "plateau"], help="Type of learning rate scheduler")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Patience for early stopping")
    parser.add_argument("--train_n_last_layers", type=int, default=0, help="Number of last layers of BERT to fine-tune")
    parser.add_argument("--inject_embedding", action="store_true", help="Inject character embeddings into the model")
    parser.add_argument("--pooling_method", type=str, default="mean", choices=["mean", "cls"], help="Pooling method for sentence embeddings")
    parser.add_argument("--retrain", action="store_true", help="Retrain the model even if saved models exist")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Pretrained model name")
    parser.add_argument("--eval_only", action="store_true", help="Run evaluation only without training")
    parser.add_argument("--sentence_mask_type", type=str, default=None, help="Masking type")
    parser.add_argument("--threshold", type=int, default=20, help="Minimum sentences per character")
    parser.add_argument("--moving_avg", action="store_true", help="Use moving average smoothing for metrics")
    parser.add_argument("--moving_avg_window", type=int, default=-1, help="Window size for moving average smoothing (-1 means no windowing)")

    args = parser.parse_args()
    main(args)
