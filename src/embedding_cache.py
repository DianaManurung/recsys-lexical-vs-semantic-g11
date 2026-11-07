"""Make sure the item embedding cache used by ``SemanticEmbeddingRecommender``
exists, generating it with the existing ``src/semantic_embedding.py``.

The cache lives in ``outputs/embeddings/`` (ignored by git). This module does
not change how embeddings are computed; when the files are missing it runs the
same steps as Lenni's ``main()`` and logs encoding time, truncation and device
to ``outputs/logs/embedding_cache_run.json``.
"""

import platform
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS_PATH = ROOT / "outputs" / "embeddings" / "item_embeddings.npy"
ITEM_IDS_PATH = ROOT / "outputs" / "embeddings" / "item_ids.npy"
LOG_PATH = ROOT / "outputs" / "logs" / "embedding_cache_run.json"


def ensure_item_embeddings():
    """Return information about the cache, encoding the items if needed."""
    info = {"cache_dir": EMBEDDINGS_PATH.parent.relative_to(ROOT).as_posix()}
    if EMBEDDINGS_PATH.exists() and ITEM_IDS_PATH.exists():
        info["from_cache"] = True
        info["encode_seconds"] = None
    else:
        from src import semantic_embedding as se

        # Same steps as se.main(), kept here so the returned stats
        # (truncated texts, max_seq_length) can be logged.
        candidate_ids = se.load_candidate_ids()
        item_text = se.load_item_text(se.DEFAULT_ITEMS_PATH, candidate_ids)
        missing = candidate_ids - set(item_text)
        if missing:
            raise ValueError(f"{len(missing)} candidate items not found in items.csv")
        t0 = time.perf_counter()
        item_ids, embeddings, stats = se.build_item_embeddings(item_text)
        info["encode_seconds"] = round(time.perf_counter() - t0, 1)
        EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.save(EMBEDDINGS_PATH, embeddings, allow_pickle=False)
        np.save(ITEM_IDS_PATH, item_ids, allow_pickle=False)
        info["from_cache"] = False
        info["model_name"] = se.MODEL_NAME
        info["batch_size"] = se.BATCH_SIZE
        info.update(stats)

    emb = np.load(EMBEDDINGS_PATH, mmap_mode="r")
    info["shape"] = list(emb.shape)
    info["device"] = _device()
    if not info["from_cache"]:
        import json
        from datetime import datetime

        log = {"created_at": datetime.now().isoformat(timespec="seconds"),
               "generated_by": "src/semantic_embedding.py (same steps as main())",
               **info, "environment": _versions()}
        LOG_PATH.write_text(json.dumps(log, indent=2), encoding="utf-8")
    return info


def _versions():
    import sentence_transformers
    import torch

    return {"python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "sentence_transformers": sentence_transformers.__version__}


def _device():
    """Describe the hardware used for encoding (the encoder runs on CPU)."""
    try:
        import torch

        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        gpu = None
    return {
        "encoder_device": "cpu",
        "cpu": platform.processor(),
        "cuda_gpu_available": gpu,
    }
