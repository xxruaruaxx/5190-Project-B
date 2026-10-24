from __future__ import annotations

import hashlib
import re
from typing import Iterable, List, Sequence

import torch
import torch.nn as nn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def _simple_tokenize(text: str) -> List[str]:
    text = text.lower()
    # replace punctuation with space, keep letters and numbers
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [t for t in text.split() if t]
    return tokens


def _hash_token(token: str, mod: int) -> int:
    # stable hashing into [0, mod)
    h = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(h, 16) % mod


class HashedTextClassifier(nn.Module):
    """
    Simple neural baseline:
    - Tokenization: regex-based
    - Vocab: hashing trick to a fixed bucket size (no stored vocab needed)
    - Architecture: EmbeddingBag -> ReLU -> Linear(num_classes=2)
    - Predicts class indices (e.g., 0/1)

    This module implements `predict(batch_of_texts)` for leaderboard compatibility.
    """

    def __init__(self, num_buckets: int = 50000, embed_dim: int = 128, hidden_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.num_buckets = num_buckets
        self.embedding = nn.EmbeddingBag(num_buckets, embed_dim, mode="mean")
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, token_ids: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        # token_ids: [N_tokens], offsets: [batch_size]
        x = self.embedding(token_ids, offsets)  # [batch_size, embed_dim]
        logits = self.fc(x)  # [batch_size, num_classes]
        return logits

    @torch.no_grad()
    def predict(self, texts: Iterable[str]) -> List[int]:
        self.eval()
        texts_list = list(texts)
        # Build flattened token id tensor and offsets for EmbeddingBag
        all_ids: List[int] = []
        offsets: List[int] = []
        running_total = 0
        for t in texts_list:
            offsets.append(running_total)
            tokens = _simple_tokenize(t)
            ids = [_hash_token(tok, self.num_buckets) for tok in tokens] or [0]
            all_ids.extend(ids)
            running_total += len(ids)
        token_ids_t = torch.tensor(all_ids, dtype=torch.long)
        offsets_t = torch.tensor(offsets, dtype=torch.long)
        logits = self.forward(token_ids_t, offsets_t)
        preds = torch.argmax(logits, dim=-1)
        return preds.cpu().tolist()


class NewsClassifier(HashedTextClassifier):
    """
    Backward-compatible name expected by the evaluator. Inherits the hashed classifier.
    """

    def __init__(self):
        super().__init__(num_buckets=50000, embed_dim=128, hidden_dim=128, num_classes=2)


def get_model() -> NewsClassifier:
    return NewsClassifier()



