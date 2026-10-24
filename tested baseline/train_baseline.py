from __future__ import annotations

import argparse
import hashlib
import os
import random
import re
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from model import NewsClassifier
from preprocess import prepare_data


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _simple_tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [t for t in text.split() if t]
    return tokens


def _hash_token(token: str, mod: int) -> int:
    h = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(h, 16) % mod


def texts_to_bag_inputs(texts: Sequence[str], num_buckets: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build inputs for EmbeddingBag from a batch of texts.
    Returns:
      - token_ids: LongTensor of shape [N_tokens_in_batch]
      - offsets: LongTensor of shape [batch_size]
    """
    all_ids: List[int] = []
    offsets: List[int] = []
    running_total = 0
    for t in texts:
        offsets.append(running_total)
        ids = [_hash_token(tok, num_buckets) for tok in _simple_tokenize(t)] or [0]
        all_ids.extend(ids)
        running_total += len(ids)
    token_ids_t = torch.tensor(all_ids, dtype=torch.long)
    offsets_t = torch.tensor(offsets, dtype=torch.long)
    return token_ids_t, offsets_t


def encode_labels(labels: Sequence[str]) -> Tuple[torch.Tensor, dict]:
    """
    Map labels to ints. Expected labels: 'FoxNews' and 'NBC'.
    """
    unique = sorted(list({str(l) for l in labels}))
    mapping = {lab: i for i, lab in enumerate(unique)}
    y = torch.tensor([mapping[str(l)] for l in labels], dtype=torch.long)
    return y, mapping


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == targets).float().sum().item()
    return correct / max(1, targets.numel())


def train(
    model: NewsClassifier,
    texts_train: Sequence[str],
    y_train: torch.Tensor,
    texts_val: Sequence[str],
    y_val: torch.Tensor,
    *,
    device: str = "cpu",
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
) -> None:
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    num_buckets = model.num_buckets

    n_train = len(texts_train)
    for epoch in range(1, epochs + 1):
        # shuffle
        indices = np.random.permutation(n_train)
        epoch_loss = 0.0
        epoch_acc = 0.0
        steps = 0
        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            idx = indices[start:end]
            batch_texts = [texts_train[i] for i in idx]
            batch_y = y_train[idx]

            token_ids, offsets = texts_to_bag_inputs(batch_texts, num_buckets)
            token_ids = token_ids.to(device)
            offsets = offsets.to(device)
            targets = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(token_ids, offsets)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                acc = accuracy_from_logits(logits, targets)

            epoch_loss += loss.item()
            epoch_acc += acc
            steps += 1

        epoch_loss /= max(1, steps)
        epoch_acc /= max(1, steps)

        # Validation
        model.eval()
        with torch.no_grad():
            val_token_ids, val_offsets = texts_to_bag_inputs(texts_val, num_buckets)
            val_logits = model(val_token_ids.to(device), val_offsets.to(device))
            val_loss = criterion(val_logits, y_val.to(device)).item()
            val_acc = accuracy_from_logits(val_logits, y_val.to(device))
        model.train()

        print(f"Epoch {epoch:02d} | train loss {epoch_loss:.4f} acc {epoch_acc:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train hashed text baseline and save model.pt")
    parser.add_argument("--csv", type=str, default="url_train16k.csv", help="Path to url_only_data.csv")
    parser.add_argument("--output", type=str, default="./model.pt", help="Output path to save trained model")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    set_seed(args.seed)

    # Load and prepare data
    texts, labels = prepare_data(args.csv)
    if len(texts) == 0:
        raise RuntimeError("No data produced by preprocess. Ensure URLs map to known outlets.")

    y_all, mapping = encode_labels(labels)
    # Prefer class order FoxNews=0, NBC=1 if present
    # (mapping is only used for training; saved model predicts indices)

    X_train, X_val, y_train, y_val = train_test_split(texts, y_all, test_size=0.2, random_state=args.seed, stratify=y_all)

    # Build model
    model = NewsClassifier()

    # Train
    train(
        model=model,
        texts_train=X_train,
        y_train=y_train,
        texts_val=X_val,
        y_val=y_val,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Save full model object for easiest loading downstream
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), "model.pt")
    print(f"Saved trained model to {args.output}")


if __name__ == "__main__":
    main()


