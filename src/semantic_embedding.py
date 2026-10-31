
"""Semantic embedding baseline using Sentence Transformers."""

from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# Project paths
ROOT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_ITEMS_PATH = ROOT_DIR / "data" / "items.csv"
DEFAULT_CANDIDATES_PATH = (
    ROOT_DIR / "data" / "candidate_items_warm.csv"
)

DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs" / "embeddings"

DEFAULT_EMBEDDINGS_PATH = (
    DEFAULT_OUTPUT_DIR / "item_embeddings.npy"
)
DEFAULT_ITEM_IDS_PATH = DEFAULT_OUTPUT_DIR / "item_ids.npy"

# Model configuration
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32
DEVICE = "cpu"
TEXT_COL = "content_text_basic"


def load_item_text(path, cand_set, text_col=TEXT_COL):
    """Load text for candidate items from items.csv."""

    items = pd.read_csv(
        path,
        dtype={"item_id": str},
        usecols=["item_id", text_col],
        encoding="utf-8-sig",
    )

    items = items.drop_duplicates("item_id", keep="first")
    items = items[items["item_id"].isin(cand_set)]

    return dict(
        zip(
            items["item_id"],
            items[text_col].fillna("").astype(str),
        )
    )


def load_candidate_ids(path=DEFAULT_CANDIDATES_PATH):
    """Load warm candidate item IDs."""

    candidates = pd.read_csv(
        path,
        dtype={"item_id": str},
        encoding="utf-8-sig",
    )

    if "item_id" not in candidates.columns:
        raise ValueError("candidate_items_warm.csv must contain item_id")

    return set(candidates["item_id"].dropna())


def count_truncated_texts(texts, tokenizer, max_seq_length):
    """Count texts and tokens exceeding the model input limit."""

    truncated_texts = 0
    truncated_tokens = 0

    for start in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[start:start + BATCH_SIZE]

        encoded = tokenizer(
            batch_texts,
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )

        for token_ids in encoded["input_ids"]:
            excess = max(0, len(token_ids) - max_seq_length)

            if excess > 0:
                truncated_texts += 1
                truncated_tokens += excess

    return truncated_texts, truncated_tokens


def build_item_embeddings(
    item_text,
    model_name=MODEL_NAME,
    batch_size=BATCH_SIZE,
    device=DEVICE,
):
    """Encode item text into normalized semantic embeddings."""

    item_ids = sorted(item_text)

    texts = [
        str(item_text[item_id]).strip()
        for item_id in item_ids
    ]

    empty_count = sum(not text for text in texts)

    if empty_count > 0:
        raise ValueError(
            f"Found {empty_count} empty item texts"
        )

    print(f"Loading model: {model_name}")

    model = SentenceTransformer(
        model_name,
        device=device,
    )

    embedding_dim = model.get_embedding_dimension()
    max_seq_length = model.max_seq_length

    # Count texts that exceed the model's input limit.
    truncated_texts, truncated_tokens = count_truncated_texts(
        texts,
        model.tokenizer,
        max_seq_length,
    )

    print("\nEncoding item text...")
    print("Number of items:", len(texts))
    print("Batch size:", batch_size)
    print("Maximum sequence length:", max_seq_length)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    # Validate embedding dimensions.
    expected_shape = (len(item_ids), embedding_dim)

    if embeddings.shape != expected_shape:
        raise ValueError(
            f"Unexpected embedding shape: {embeddings.shape}; "
            f"expected {expected_shape}"
        )

    # Validate L2 normalization.
    norms = np.linalg.norm(embeddings, axis=1)

    if not np.allclose(norms, 1.0, atol=1e-5):
        raise ValueError("L2 normalization validation failed")

    return (
        np.asarray(item_ids, dtype=str),
        embeddings,
        {
            "embedding_dimension": embedding_dim,
            "max_seq_length": max_seq_length,
            "truncated_texts": truncated_texts,
            "truncated_tokens": truncated_tokens,
        },
    )


def main():
    """Generate and save semantic embeddings for warm candidates."""

    print("Loading candidate item IDs...")

    candidate_ids = load_candidate_ids()

    print("Candidate items:", len(candidate_ids))

    print("\nLoading item text...")

    item_text = load_item_text(
        DEFAULT_ITEMS_PATH,
        candidate_ids,
    )

    missing_ids = candidate_ids - set(item_text)

    if missing_ids:
        raise ValueError(
            f"{len(missing_ids)} candidate items not found in items.csv"
        )

    print("Items with text:", len(item_text))

    item_ids, embeddings, stats = build_item_embeddings(
        item_text
    )

    # Save generated data locally.
    DEFAULT_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        DEFAULT_EMBEDDINGS_PATH,
        embeddings,
        allow_pickle=False,
    )

    np.save(
        DEFAULT_ITEM_IDS_PATH,
        item_ids,
        allow_pickle=False,
    )

    print("\n=== SEMANTIC EMBEDDING RESULTS ===")
    print("Number of items:", len(item_ids))
    print("Embedding shape:", embeddings.shape)
    print("Embedding dimension:", stats["embedding_dimension"])
    print("Maximum sequence length:", stats["max_seq_length"])
    print("Truncated texts:", stats["truncated_texts"])
    print("Truncated tokens:", stats["truncated_tokens"])
    print("Minimum vector norm:", float(norms_min(embeddings)))
    print("Maximum vector norm:", float(norms_max(embeddings)))

    print("\nSaved files:")
    print(DEFAULT_EMBEDDINGS_PATH)
    print(DEFAULT_ITEM_IDS_PATH)


def norms_min(embeddings):
    return np.linalg.norm(embeddings, axis=1).min()


def norms_max(embeddings):
    return np.linalg.norm(embeddings, axis=1).max()


if __name__ == "__main__":
    main()
