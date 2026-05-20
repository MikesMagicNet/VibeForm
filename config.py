# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  config.py  –  Single source of truth for ALL project settings             ║
# ║                                                                            ║
# ║  Edit this file to change anything about training, data, model size, etc.  ║
# ║  Every other file imports from here — you never need to hunt for settings. ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


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
# ══════════════════════════════════════════════════════════════════════════════

PATHS = {
    "vocab": "vocab.json",            # ... tokenizer vocabulary
    "checkpoint": "checkpoint.pt",    # ... trained model weights
    "metrics": "metrics.json",        # ... dashboard metrics (auto-generated)
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
