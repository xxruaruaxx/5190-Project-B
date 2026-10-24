from __future__ import annotations

import csv
import html
import os
from typing import List, Tuple
from urllib.parse import urlparse, unquote


def _slug_to_text(path: str) -> str:
    # Take last segments, split on -, _, and decode
    segments = [seg for seg in path.split("/") if seg]
    if not segments:
        return ""
    tail = segments[-1]
    tail = tail.split("?")[0]
    tail = tail.split("#")[0]
    tail = tail.rsplit(".", 1)[0]  # drop extension
    # Replace separators with spaces and unescape
    text = tail.replace("-", " ").replace("_", " ")
    text = unquote(html.unescape(text))
    return text.strip()


def _label_from_host(netloc: str) -> str:
    host = netloc.lower()
    if "foxnews" in host:
        return "FoxNews"
    if "nbcnews" in host or "nbc.com" in host or "nbc" in host:
        return "NBC"
    return ""


def _process(url: str, texts: List[str], labels: List[str]) -> None:
    parsed = urlparse(url)
    label = _label_from_host(parsed.netloc)
    # Skip links from other websites. Links WITHOUT a website name (how the leaderboard
    # gives them to you) are kept, with an empty label, so every row gets a prediction.
    if parsed.netloc and not label:
        return
    text = _slug_to_text(parsed.path)
    if not text:
        text = parsed.netloc
    texts.append(text)
    labels.append(label)


def prepare_data(csv_path: str) -> Tuple[List[str], List[str]]:
    """
    Reads a CSV with a single 'url' column and produces:
      - X: list of text strings derived from URL slugs
      - y: list of labels inferred from host ('FoxNews' or 'NBC'), or '' if the
           link has no website name (the leaderboard removes it from test links)
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    texts: List[str] = []
    labels: List[str] = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        # If no header named 'url', try to handle a single-column CSV
        if "url" not in (reader.fieldnames or []):
            f.seek(0)
            for row in csv.reader(f):
                if row:
                    _process(row[0], texts, labels)
        else:
            for row in reader:
                _process(row.get("url", "") or "", texts, labels)

    return texts, labels


if __name__ == "__main__":
    # Example usage
    input_csv = "url_train.csv"
    X, y = prepare_data(input_csv)
