# IMPORT STATEMENTS
import numpy as np
import matplotlib.pyplot as plt
import json
import os

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import AdamW
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, f1_score

# Language Models
from transformers import BertTokenizer, BertForMaskedLM, BertModel, AutoTokenizer, AutoModel, get_scheduler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from models import MoralRelevanceDataset, Autoencoder, MoralClassifier
import argparse, time

# Load the data for all models
SENTENCE_DATA_PATH = '../data/dump/project_6_data.json'
input_dir = None


def custom_collate_fn(batch):
    """
    Custom collate function to handle the batch data structure.

    This resolves the issues of different past_sentences lengths and ensures that the data is properly collated into a batch.
    """

    batch_collated = {}

    for key in batch[0]:
        if key == "past_sentences":
            # Keep as list of lists of strings
            batch_collated[key] = [item[key] for item in batch]
        else:
            batch_collated[key] = torch.utils.data.default_collate([item[key] for item in batch])

    return batch_collated

def update_char_vec_embeddings(dataset, tokenizer, encoder, device, batch_size=64, pooling_method = 'mean'):
    """
    Update 'avg_embedding' in-place for each example in the dataset based on current encoder state.

    Here we will apply an algorithm that makes sure there's no duplication of effort during embeddings generation.

    Steps:
    - Generate the sentence embeddings for all the characters
    - Index them properly, take the average, then assign it to corresponding data point in original dataset

    Some variations:
    - Since there will be data points splitting and a character's data points might be separated between test and training dataset,
      we need to make sure we assign them properly and by order.

    """
    encoder.eval()

    with torch.no_grad():
        cur_char = None
        cur_movie = None
        first_index = 0
        last_index = 0
        data_length = len(dataset)
        for i, row in enumerate(tqdm(dataset, desc="Updating char_vecs")):

            cur_movie = row["movie"]
            cur_char = row["character"]

            # Check if the next character is the same as current character
            # Also check if the next data point is not available
            if i + 1 < data_length and dataset[i+1]["movie"] == cur_movie and dataset[i+1]["character"] == cur_char:
                last_index += 1
                continue

            past_sentences = row.get("past_sentences", [])
            num_past_sentences = len(past_sentences)

            ### TODO: EDIT THE BELOW

            if num_past_sentences == 0:
                row["avg_embedding"] = torch.zeros(768)
                continue

            # Encode in chunks to save memory
            all_embeddings = []
            for j in range(0, num_past_sentences, batch_size):
                batch_sentences = past_sentences[j:j + batch_size]
                encoded = tokenizer(
                    batch_sentences,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256
                ).to(device)

                outputs = encoder(**encoded)
                last_hidden = outputs.last_hidden_state
                if pooling_method == 'mean':
                    attn_mask = encoded["attention_mask"].unsqueeze(-1)  # shape: (batch_size, seq_len, 1)
                    # .clamp(min = 1e-9) prevents division by 0
                    pooled = (last_hidden * attn_mask).sum(1) / attn_mask.sum(1).clamp(min=1e-9)
                elif pooling_method == 'cls':
                    pooled = last_hidden[:, 0, :]

                all_embeddings.append(pooled.cpu())

            # all_embeddings contains all sentence embeddings that we need
            # Start the indexing logic + averaging

            flat_embeds = torch.cat(all_embeddings, dim=0)  # Shape: (num_sentences, 768)

            for k in range(num_past_sentences-1):   # We use num_past_sentences-1 because there needs to be at least 1 past sentence for each data point

                # If we reach the left edge
                # This is the case where a character data points are splited by train-test split
                if i - k < 0:
                    break

                cur_past_sentences = flat_embeds[:len(flat_embeds) - k]
                full_embed = cur_past_sentences.mean(dim=0)

                dataset[last_index-k]["avg_embedding"] = full_embed

            # We update the first_index
            first_index = i+1
            last_index = first_index


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

def train_moral_classifier(
    train_dataset, val_dataset, model_name,
    use_vae=False, use_one_hot=False, char2id=None, train_n_last_layers=0,
    latent_dim=20, alpha=0.1, beta=0.1, num_epochs=10, batch_size=32, lr_1 = 1e-3, lr_2=2e-5,
    dropout_rate=0.1, clip_grad_norm=5.0, weight_decay=1e-5,
    scheduler_type="cosine", log_path = None, pos_weight = 1.0, early_stopping_patience=3, early_stopping_metric="f1_score", minimize_metric=False,
    inject_embedding=True, classification_pooling_method = "cls", injection_pooling_method = "mean", injection_method = "sum"
):
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name)

    for name, param in base_model.named_parameters():
        if any(f"encoder.layer.{i}" in name for i in range(12-train_n_last_layers, 12)) or "pooler" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model.to(device)

    model_H = Autoencoder(768, latent_dim, dropout=dropout_rate)
    model_H.to(device)

    classifier = MoralClassifier(base_model, inject_operation=injection_method, inference_pooling_method=classification_pooling_method).to(device)

    # Character embedding setup
    if use_one_hot:
        if char2id is None:
            raise ValueError("char2id must be provided when use_one_hot is True")
        character_embedding = nn.Embedding(len(char2id), 768).to(device)
        embedding_params = list(character_embedding.parameters())
    else:
        character_embedding = None
        embedding_params = []

    optimizer = AdamW([
        {"params": classifier.classifier.parameters(), "lr": lr_1},  # classification head
        {"params": model_H.parameters(), "lr": lr_1},                # autoencoder
        {"params": [p for n, p in classifier.bert.named_parameters() if p.requires_grad], "lr": lr_2},  # BERT last 4
    ], weight_decay=weight_decay)

    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device))

    if scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    elif scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    elif scheduler_type == "cosine":
        num_training_steps = len(loader) * num_epochs
        num_warmup_steps = int(0.1 * num_training_steps)
        scheduler = get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    else:
        scheduler = None

    best_score = None
    best_epoch = -1
    patience_counter = 0
    best_state = None

    for epoch in range(num_epochs):
        model_H.train()
        classifier.train()
        total_loss, recon_total, kl_total, cls_total, ort_total = 0, 0, 0, 0, 0

        for batch_idx, batch in enumerate(tqdm(loader, desc=f"[{model_name}] Epoch {epoch+1}")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            labels = labels.float()

            if inject_embedding:
                if use_one_hot:
                    char_id = batch["character_id"].to(device)
                    char_vec = character_embedding(char_id)
                else:
                    char_vec = batch["avg_embedding"].to(device)

                if use_vae:
                    recon_vec, mu, logvar, z = model_H(char_vec)
                    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / char_vec.size(0)
                else:
                    recon_vec, z = model_H(char_vec)
                    kl_div = 0

                logits = classifier(input_ids, attention_mask, recon_vec)
            else:
                recon_vec, z, kl_div = None, None, 0
                logits = classifier(input_ids, attention_mask, None)

            logits = logits.squeeze(-1)
            cls_loss = criterion(logits, labels)

            if inject_embedding:
                recon_loss = nn.MSELoss()(recon_vec, char_vec)
                if batch_idx % 4 == 0:
                    ortho_loss = compute_orthogonality_loss(z, strategy="off_diag")
                else:
                    ortho_loss = torch.tensor(0.0, device=z.device)
            else:
                recon_loss = torch.tensor(0.0, device=device)
                ortho_loss = torch.tensor(0.0, device=device)

            # Total loss
            loss = alpha * cls_loss + recon_loss + (alpha * kl_div if use_vae else 0) + beta * ortho_loss

            optimizer.zero_grad()
            loss.backward()
            if clip_grad_norm:
                nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]], clip_grad_norm)

            # Track loss
            total_loss += loss.item()
            recon_total += recon_loss.item()
            cls_total += (alpha * cls_loss).item()
            ort_total += (beta * ortho_loss).item()
            kl_total += kl_div.item() if use_vae else 0

            optimizer.step()

        if scheduler_type == "cosine" or scheduler_type == "step":
            scheduler.step()
        elif scheduler_type == "plateau":
            scheduler.step(eval_metrics["f1_score"])  # or validation loss if tracked

        # Evaluate
        eval_metrics = evaluate_classifier(model_H, classifier, val_dataset, tokenizer, use_one_hot, character_embedding, inject_embedding = inject_embedding)
        # We might not need to keep track of the metrics for training data (Loss might be enough)
        train_metrics = evaluate_classifier(model_H, classifier, train_dataset, tokenizer, use_one_hot, character_embedding, inject_embedding = inject_embedding)

        # Get metric to track
        val_score = eval_metrics[early_stopping_metric]
        if minimize_metric:
            improved = best_score is None or val_score < best_score
        else:
            improved = best_score is None or val_score > best_score

        # Check if we should stop now
        if improved:
            best_score = val_score
            best_epoch = epoch
            patience_counter = 0
            best_state = {
                "model_H": model_H.state_dict(),
                "classifier": classifier.state_dict(),
                "character_embedding": character_embedding.state_dict() if character_embedding else None,
            }
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1} — best {early_stopping_metric}: {best_score:.4f} (epoch {best_epoch+1})")
                break

        # Print the performance metrics
        print(f"Epoch {epoch+1} [{model_name}]:")
        print(f"  Train Total Loss={total_loss:.4f}, Recon={recon_total:.4f}, ClassLoss={cls_total:.4f}, Ortho loss: {ort_total:.4f}" +
              (f", KL={kl_total:.4f}" if use_vae else "") + f" Accuracy={train_metrics['accuracy']:.4f}, F1={train_metrics['f1_score']:.4f}, ROC AUC={train_metrics['roc_auc']:.4f}, Negative-F1={train_metrics['negative_class_f1_score']:.4f}")
        print(f"  Eval: Accuracy={eval_metrics['accuracy']:.4f}, Precision={eval_metrics['precision']:.4f}, Recall={eval_metrics['recall']:.4f}, F1={eval_metrics['f1_score']:.4f}, ROC AUC={eval_metrics['roc_auc']:.4f}, Negative-F1={eval_metrics['negative_class_f1_score']:.4f}")

        # Log the performance metrics
        if log_path:
            log_metrics_to_csv(
                log_path,
                model_name=model_name,
                epoch=epoch + 1,
                train_losses={
                    "total": total_loss,
                    "recon": recon_total,
                    "cls": cls_total,
                    "kl": kl_total,
                    "ort": ort_total
                },
                train_metrics=train_metrics,
                eval_metrics=eval_metrics
            )

        if train_n_last_layers > 0:
            print(f"\n[Epoch {epoch+1}] Recomputing char_vec embeddings...")
            update_char_vec_embeddings(train_dataset, tokenizer, classifier.bert, device, pooling_method=injection_pooling_method)
            update_char_vec_embeddings(val_dataset, tokenizer, classifier.bert, device, pooling_method=injection_pooling_method)

    # Restore best model
    if best_state:
        model_H.load_state_dict(best_state["model_H"])
        classifier.load_state_dict(best_state["classifier"])
        if character_embedding and best_state["character_embedding"]:
            character_embedding.load_state_dict(best_state["character_embedding"])

    return model_H, classifier, character_embedding

import csv
import os

def init_csv(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)  # Create parent directories
    header = [
        "model", "epoch",
        "train_total_loss", "train_recon_loss", "train_cls_loss", "train_kl_loss", "train_ort_loss",
        "train_accuracy", "train_f1", "train_roc_auc", "train_negative_f1",
        "eval_accuracy", "eval_precision", "eval_recall", "eval_f1", "eval_roc_auc",
        "eval_negative_f1"
    ]
    new_file = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(header)


def log_metrics_to_csv(log_path, model_name, epoch, train_losses, train_metrics, eval_metrics):
    row = [
        model_name, epoch,
        train_losses["total"], train_losses["recon"], train_losses["cls"], train_losses["kl"], train_losses["ort"],
        train_metrics["accuracy"], train_metrics["f1_score"], train_metrics["roc_auc"], train_metrics["negative_class_f1_score"],
        eval_metrics["accuracy"], eval_metrics["precision"], eval_metrics["recall"],
        eval_metrics["f1_score"], eval_metrics["roc_auc"], eval_metrics["negative_class_f1_score"]
    ]
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def evaluate_classifier(model_H, classifier, dataset, tokenizer, use_one_hot=False, character_embedding=None, batch_size = 32, inject_embedding=True,
                        classification_pooling_method = "cls", injection_pooling_method = "mean"):

    # We do not need to use of the classification_pooling_method, injection_method, and injection_pooling_method:
    # classification_pooling_method depends on the base model and is handled inside the classifier
    # injection_method is handled in the classifier
    # injection_pooling_method is handled during the generation of avg_embedding in the data preprocessing step

    model_H.eval()
    classifier.eval()
    if character_embedding:
        character_embedding.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_H.to(device)
    classifier.to(device)
    if character_embedding:
        character_embedding.to(device)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0.0
    criterion = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device).float()

            if inject_embedding:
                if use_one_hot:
                    char_id = batch["character_id"].to(device)
                    char_vec = character_embedding(char_id)
                else:
                      char_vec = batch["avg_embedding"].to(device)

                recon_vec, *_ = model_H(char_vec)
                logits = classifier(input_ids=input_ids, attention_mask=attention_mask, char_vec=recon_vec)
            else:
                logits = classifier(input_ids=input_ids, attention_mask=attention_mask, char_vec=None)

            logits = logits.squeeze(-1)

            loss = criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)  # For binary classification
            preds = (probs >= 0.5).long()  # Thresholding

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    f1_neg = f1_score(all_labels, all_preds, pos_label=0)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary", zero_division=0)
    roc_auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0

    return {
        "loss": total_loss / len(loader),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "negative_class_f1_score": f1_neg
    }

model_names = {
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
    "deberta": "microsoft/deberta-v3-base",
    "electra": "google/electra-base-discriminator"
    }

import argparse

def main(args):
    """
    Main function to train a moral classification model and save the trained components.
    Args:
        args (argparse.Namespace): A namespace object containing the following attributes:
            - input_dir (str): Path to the input directory containing training and testing data.
            - model_name (str): Name of the model to be used for training.
            - log_path (str): Path to the log file for recording training progress and metrics.
            - use_vae (bool): Whether to use a Variational Autoencoder (VAE) in the model.
            - use_one_hot (bool): Whether to use one-hot encoding for character embeddings.
            - train_n_last_layers (int): Number of last layers to fine-tune during training.
            - latent_dim (int): Dimensionality of the latent space for the VAE.
            - alpha (float): Weight for the reconstruction loss in the VAE.
            - beta (float): Weight for the KL divergence loss in the VAE.
            - num_epochs (int): Number of training epochs.
            - batch_size (int): Batch size for training.
            - lr_1 (float): Learning rate for the first optimizer.
            - lr_2 (float): Learning rate for the second optimizer.
            - dropout_rate (float): Dropout rate to be applied in the model.
            - clip_grad_norm (float): Maximum norm for gradient clipping.
            - weight_decay (float): Weight decay (L2 regularization) for the optimizer.
            - scheduler_type (str): Type of learning rate scheduler to use.
            - pos_weight (float): Weight for positive samples in the loss function.
            - early_stopping_patience (int): Number of epochs to wait for improvement before early stopping.
            - early_stopping_metric (str): Metric to monitor for early stopping.
            - minimize_metric (bool): Whether to minimize the early stopping metric.
            - inject_embedding (bool): Whether to inject character embeddings into the model.
            - classification_pooling_method (str): Pooling method for classification (e.g., "mean", "max").
            - injection_pooling_method (str): Pooling method for embedding injection (e.g., "mean", "max").
            - injection_method (str): Method for injecting embeddings into the model.
    Workflow:
        1. Reads training and testing data from the specified input directory.
        2. Initializes a CSV log file for recording training progress.
        3. Creates training and validation datasets using the `MoralRelevanceDataset` class.
        4. Trains the moral classification model using the `train_moral_classifier` function.
        5. Saves the trained model components (`model_H`, `classifier`, and `character_embedding`) to the output directory.
    Outputs:
        - Trained model components are saved in the `models/moral_classifier` directory:
            - `model_H.pth`: The trained model_H state dictionary.
            - `classifier.pth`: The trained classifier state dictionary.
            - `character_embedding.pth`: The trained character embedding state dictionary (if applicable).
    """

    input_dir = args.input_dir
    model_name = args.model_name
    use_vae = args.use_vae
    use_one_hot = args.use_one_hot
    train_n_last_layers = args.train_n_last_layers
    latent_dim = args.latent_dim
    alpha = args.alpha
    beta = args.beta
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    lr_1 = args.lr_1
    lr_2 = args.lr_2
    dropout_rate = args.dropout_rate
    clip_grad_norm = args.clip_grad_norm
    weight_decay = args.weight_decay
    scheduler_type = args.scheduler_type
    pos_weight = args.pos_weight
    early_stopping_patience = args.early_stopping_patience
    early_stopping_metric = args.early_stopping_metric
    minimize_metric = args.minimize_metric
    inject_embedding = args.inject_embedding
    classification_pooling_method = args.classification_pooling_method
    injection_pooling_method = args.injection_pooling_method
    injection_method = args.injection_method

    model_H_path = os.path.join(args.output_dir, "model_H.pth")
    classifier_path = os.path.join(args.output_dir, "classifier.pth")
    log_path = os.path.join(args.output_dir, "logs", f"{args.model_name}_moral_classification_log.csv")

    if args.retrain and os.path.exists(model_H_path) and os.path.exists(classifier_path) and os.path.exists(log_path):
        print(f"Output directory {args.output_dir} already exists. Proceeding to retrain and overwrite existing models...")
    else:
        if os.path.exists(model_H_path) and os.path.exists(classifier_path) and os.path.exists(log_path):
            print("All required files (model_H, classifier, and log) are present. Skipping retrain as requested.")
            return
        else:
            if not args.retrain:
                print("Required files (model_H, classifier, or log) are missing. Forcing retrain...")

    # We document all the arguments used for training in a text file
    arg_log_txt_file = os.path.join(args.output_dir, f"{model_name}_moral_classification_args.txt")
    with open(arg_log_txt_file, "w") as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

    suffix = injection_pooling_method
    sampling_strategy = args.sampling_strategy

    if sampling_strategy == "down":
        suffix += f"_downsampled"
    elif sampling_strategy == "up":
        suffix += f"_upsampled"
    else:
        suffix += f"_regular"

    for r in range(1, args.repeat + 1):
        curr_suffix = suffix
        if sampling_strategy == "down" or sampling_strategy == "up":
            curr_suffix += f"_split{r}"
        with open(os.path.join(input_dir, f"train_data_{curr_suffix}.json"), "r") as f:
            train_data = json.load(f)

        if sampling_strategy == "down" or sampling_strategy == "up":
            # When we do more than 1 repetition, the test data will still be the same for all repetitions
            with open(os.path.join(input_dir, f"test_data_{suffix}_split1.json"), "r") as f:
                test_data = json.load(f)
        else:
            with open(os.path.join(input_dir, f"test_data_{curr_suffix}.json"), "r") as f:
                test_data = json.load(f)

        ## NOTE: We convert the labels to 0 and 1 here since the dataset class expects numerical labels
        for k in train_data:
            if k["label"] == "Yes":
                k["label"] = 1
            else:
                k["label"] = 0

        for k in test_data:
            if k["label"] == "Yes":
                k["label"] = 1
            else:
                k["label"] = 0       

        log_path = os.path.join(args.output_dir, "logs", f"{curr_suffix}_moral_classification_log.csv")
        init_csv(log_path)

        train_dataset = MoralRelevanceDataset(train_data, tokenizer=None, use_one_hot=False)
        val_dataset = MoralRelevanceDataset(test_data, tokenizer=None, use_one_hot=False)

        model_H, classifier, character_embedding = train_moral_classifier(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            model_name=model_name,
            use_vae=use_vae,
            use_one_hot=use_one_hot,
            char2id=None,
            train_n_last_layers=train_n_last_layers,
            latent_dim=latent_dim,
            alpha=alpha,
            beta=beta,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr_1=lr_1,
            lr_2=lr_2,
            dropout_rate=dropout_rate,
            clip_grad_norm=clip_grad_norm,
            weight_decay=weight_decay,
            scheduler_type=scheduler_type,
            log_path=log_path,
            pos_weight=pos_weight,
            early_stopping_patience=early_stopping_patience,
            early_stopping_metric=early_stopping_metric,
            minimize_metric=minimize_metric,
            inject_embedding=inject_embedding,
            injection_pooling_method=injection_pooling_method,
            classification_pooling_method=classification_pooling_method,
            injection_method=injection_method
        )

        # Save the trained models
        output_dir = args.output_dir

        os.makedirs(output_dir, exist_ok=True)

        model_H_save_path = os.path.join(output_dir, f"model_H_{curr_suffix}.pth")
        classifier_save_path = os.path.join(output_dir, f"classifier_{curr_suffix}.pth")
        character_embedding_save_path = os.path.join(output_dir, f"character_embedding_{curr_suffix}.pth")

        if model_H:
            torch.save(model_H.state_dict(), model_H_save_path)

        torch.save(classifier.state_dict(), classifier_save_path)
        
        if character_embedding:
            torch.save(character_embedding.state_dict(), character_embedding_save_path)

if __name__ == "__main__":
    print("Starting moral classification training script...")
    parser = argparse.ArgumentParser(description="Train Moral Classifier")

    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing the input JSON files")
    parser.add_argument("--output_dir", type=str, default="models/moral_classifier", help="Directory to save the trained models")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Pre-trained model name")
    parser.add_argument("--use_vae", action="store_true", help="Use Variational Autoencoder for character embeddings")
    parser.add_argument("--use_one_hot", action="store_true", help="Use one-hot encoding for character embeddings")
    parser.add_argument("--train_n_last_layers", type=int, default=4, help="Number of last layers of the model to train")
    parser.add_argument("--latent_dim", type=int, default=20, help="Latent dimension for the autoencoder")
    parser.add_argument("--alpha", type=float, default=0.1, help="Weight for classification loss")
    parser.add_argument("--beta", type=float, default=0.01, help="Weight for orthogonality loss")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr_1", type=float, default=1e-3, help="Learning rate for classification head and autoencoder")
    parser.add_argument("--lr_2", type=float, default=2e-5, help="Learning rate for the pre-trained model")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate for the autoencoder")
    parser.add_argument("--clip_grad_norm", type=float, default=5.0, help="Gradient clipping norm value")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for the optimizer")
    parser.add_argument("--scheduler_type", type=str, default="cosine", choices=["cosine", "step", "plateau"], help="Type of learning rate scheduler")
    parser.add_argument("--pos_weight", type=float, default=4.0, help="Positive class weight for the loss function")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Patience for early stopping")
    parser.add_argument("--early_stopping_metric", type=str, default="f1_score", help="Metric to monitor for early stopping")
    parser.add_argument("--minimize_metric", action="store_true", help="Whether to minimize the early stopping metric")
    parser.add_argument("--inject_embedding", action="store_true", help="Inject character embeddings into the classifier")
    parser.add_argument("--classification_pooling_method", type=str, default="cls", choices=["cls", "mean"], help="Pooling method for classification")
    parser.add_argument("--injection_pooling_method", type=str, default="mean", choices=["mean", "max"], help="Pooling method for injection")
    parser.add_argument("--injection_method", type=str, default="sum", choices=["sum", "concat"], help="Method to inject embeddings")
    parser.add_argument("--retrain", action="store_true", help="Regenerate embeddings even if they already exist.")
    parser.add_argument("--sampling_strategy", type=str, default="none", choices=["down", "up", "none"], help="Sampling strategy for training data")
    parser.add_argument("--repeat", type=int, default=1, help="Number of times to repeat training with different data splits")
    args = parser.parse_args()
    main(args)
    print("Training complete.")


