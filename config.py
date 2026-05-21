# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  config.py  –  Single source of truth for ALL project settings             ║
# ║                                                                            ║
# ║  Edit this file to change anything about training, data, model size, etc.  ║
# ║  Every other file imports from here — you never need to hunt for settings. ║
# ║                                                                            ║
# ║  SECURITY: All file paths are sanitized to prevent path traversal.         ║
# ║  VALIDATION: Runtime checks ensure config values are sane before training. ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import os
import sys

# ── Path Security ─────────────────────────────────────────────────────────────
# All paths are resolved relative to the project root and validated
# to prevent path-traversal attacks (e.g., "../../etc/passwd").

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _safe_path(relative_path: str) -> str:
    """
    Resolves a relative path against the project root and ensures
    the result stays within the project directory.

    Raises:
        ValueError: If the resolved path escapes the project root.
    """
    resolved = os.path.normpath(os.path.join(_PROJECT_ROOT, relative_path))
    if not resolved.startswith(_PROJECT_ROOT):
        raise ValueError(
            f"Path traversal detected: '{relative_path}' resolves to "
            f"'{resolved}', which is outside the project root '{_PROJECT_ROOT}'."
        )
    return resolved


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL ARCHITECTURE
#  These define the size/capacity of the Transformer.
#  Changing these creates a NEW model — old checkpoints won't be compatible.
# ══════════════════════════════════════════════════════════════════════════════

MODEL = {
    "dModel": 256,      # ... embedding dimension (paper=512, smaller = faster)
    "dFF": 1024,         # ... feed-forward hidden size (paper=2048, typically 4x dModel)
    "h": 8,             # ... number of attention heads (must divide dModel evenly)
    "N": 4,             # ... number of encoder & decoder layers (paper=6, was 2)
    "dropout": 0.1,     # ... dropout probability (0.1 = drop 10% of connections)
    "seqLength": 512,    # ... max tokens per sequence (longer for reasoning text)
}


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING
#  These control HOW the model learns. Safe to change between runs.
# ══════════════════════════════════════════════════════════════════════════════

TRAINING = {
    "epochs": 50,            # ... number of full passes through the data
    "batchSize": 10,        # ... samples per training step
    "maxLR": 5e-4,          # ... peak learning rate (reached at end of warmup)
    "minLR": 1e-5,          # ... floor learning rate (end of cosine decay)
    "warmupSteps": 500,     # ... more warmup for deeper model (prevents early instability)
    "labelSmoothing": 0.2,  # ... prevents overconfidence (from the paper)
    "gradClipNorm": 1.1,    # ... max gradient norm (prevents exploding gradients)
    "logInterval": 20,      # ... print + save metrics every N batches (5=verbose, 25=normal, 100=quiet)
    "accumSteps": 4,        # ... gradient accumulation (effective batch = batchSize × accumSteps = 140)
}


# ══════════════════════════════════════════════════════════════════════════════
#  DATA
#  Controls what data the model trains on.
# ══════════════════════════════════════════════════════════════════════════════

DATA = {
    "source": "claude-opus",          # ... "wikipedia" or "claude-opus"
    "numArticles": 20000,              # ... cap to 20,000 for math problems
    "valSplit": 0.1,                  # ... fraction reserved for validation (0.1 = 10%)
    # Wikipedia settings (used when source = "wikipedia")
    "wikiDataset": "wikimedia/wikipedia",
    "wikiConfig": "20231101.en",
    # HuggingFace dataset (used when source = "claude-opus")
    "claudeDataset": "math-extraction-comp/deepseek-ai__deepseek-llm-67b-chat",
}


# ══════════════════════════════════════════════════════════════════════════════
#  TOKENIZER
#  Controls vocabulary building. Rebuild vocab.json after changing these.
# ══════════════════════════════════════════════════════════════════════════════

TOKENIZER = {
    "minFrequency": 5,          # ... minimum word count (5 for 52k-row dataset)
    "vocabArticles": 52000,     # ... rows to scan for vocabulary building
    "seqLengthForTest": 512,    # ... sequence length used during tokenizer test
}


# ══════════════════════════════════════════════════════════════════════════════
#  PATHS
#  Where files get saved. All relative to the project root.
#  Paths are sanitized via _safe_path() to prevent path traversal.
# ══════════════════════════════════════════════════════════════════════════════

PATHS = {
    "vocab": _safe_path("vocab.json"),
    "checkpoint": _safe_path("checkpoint.pt"),
    "metrics": _safe_path("metrics.json"),
}


# ══════════════════════════════════════════════════════════════════════════════
#  RESUME / FINE-TUNE
#  Controls whether training starts fresh or continues from a checkpoint.
# ══════════════════════════════════════════════════════════════════════════════

RESUME = {
    "enabled": True,      # ... False = train from scratch with new dataset
                            # ... True = load checkpoint if it exists
                            # ... Set to False + delete checkpoint.pt for a clean start

    "loadOptimizer": False, # ... True = resume optimizer state (same data, more epochs)
                            # ... False = fresh optimizer (fine-tuning on new/different data)
}


# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

DASHBOARD = {
    "port": 8080,           # ... http://localhost:8080
    "pollInterval": 2000,   # ... ms between metric updates
}


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG VALIDATION
#  Catches misconfigurations before they cause cryptic errors at runtime.
# ══════════════════════════════════════════════════════════════════════════════

def validate_config():
    """
    Validates all configuration values and raises clear errors for
    invalid settings. Call this at the start of train.py / generate.py.

    Returns:
        list[str]: Warning messages (non-fatal issues).

    Raises:
        ValueError: For invalid configuration values.
    """
    warnings = []

    # ── Model validation ──────────────────────────────────────────────────
    if MODEL["dModel"] % MODEL["h"] != 0:
        raise ValueError(
            f"dModel ({MODEL['dModel']}) must be divisible by h ({MODEL['h']}). "
            f"Each attention head gets dModel/h = {MODEL['dModel']/MODEL['h']:.1f} dims."
        )

    if MODEL["dModel"] <= 0:
        raise ValueError(f"dModel must be positive, got {MODEL['dModel']}")

    if MODEL["dFF"] <= 0:
        raise ValueError(f"dFF must be positive, got {MODEL['dFF']}")

    if MODEL["N"] <= 0:
        raise ValueError(f"N (num layers) must be positive, got {MODEL['N']}")

    if not (0.0 <= MODEL["dropout"] < 1.0):
        raise ValueError(f"dropout must be in [0, 1), got {MODEL['dropout']}")

    if MODEL["seqLength"] <= 0:
        raise ValueError(f"seqLength must be positive, got {MODEL['seqLength']}")

    # ── Training validation ───────────────────────────────────────────────
    if TRAINING["epochs"] <= 0:
        raise ValueError(f"epochs must be positive, got {TRAINING['epochs']}")

    if TRAINING["batchSize"] <= 0:
        raise ValueError(f"batchSize must be positive, got {TRAINING['batchSize']}")

    if TRAINING["maxLR"] <= 0:
        raise ValueError(f"maxLR must be positive, got {TRAINING['maxLR']}")

    if TRAINING["minLR"] < 0:
        raise ValueError(f"minLR must be non-negative, got {TRAINING['minLR']}")

    if TRAINING["minLR"] >= TRAINING["maxLR"]:
        warnings.append(
            f"minLR ({TRAINING['minLR']}) >= maxLR ({TRAINING['maxLR']}). "
            f"Cosine decay will be flat."
        )

    if not (0.0 <= TRAINING["labelSmoothing"] < 1.0):
        raise ValueError(f"labelSmoothing must be in [0, 1), got {TRAINING['labelSmoothing']}")

    if TRAINING["gradClipNorm"] <= 0:
        raise ValueError(f"gradClipNorm must be positive, got {TRAINING['gradClipNorm']}")

    if TRAINING["accumSteps"] <= 0:
        raise ValueError(f"accumSteps must be positive, got {TRAINING['accumSteps']}")

    # ── Data validation ───────────────────────────────────────────────────
    if DATA["source"] not in ("wikipedia", "claude-opus"):
        raise ValueError(f"data source must be 'wikipedia' or 'claude-opus', got '{DATA['source']}'")

    if not (0.0 < DATA["valSplit"] < 1.0):
        raise ValueError(f"valSplit must be in (0, 1), got {DATA['valSplit']}")

    if DATA["numArticles"] <= 0:
        raise ValueError(f"numArticles must be positive, got {DATA['numArticles']}")

    # ── Tokenizer validation ──────────────────────────────────────────────
    if TOKENIZER["minFrequency"] <= 0:
        raise ValueError(f"minFrequency must be positive, got {TOKENIZER['minFrequency']}")

    return warnings
