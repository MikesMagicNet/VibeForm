# Transformer

A from-scratch Transformer implementation in PyTorch — complete with training, generation, a live dashboard, and a configurable tokenizer inspired by modern tokenizer architectures.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                    config.py                         │
│         Single source of truth for ALL settings      │
└───────┬──────────┬──────────┬───────────┬────────────┘
        │          │          │           │
   ┌────▼───┐ ┌───▼────┐ ┌──▼──┐  ┌─────▼──────┐
   │tokenizer│ │ model  │ │train│  │  dashboard  │
   │   .py   │ │  .py   │ │ .py │  │  .py/.html  │
   └────┬────┘ └───┬────┘ └──┬──┘  └─────┬──────┘
        │          │         │            │
        │     Embedding      │        HTTP API
        │     + Attention    │     (metrics, tokenizer
        │     + FFN layers   │      config, viz)
        │          │         │
   ┌────▼──────────▼─────────▼────┐
   │        generate.py           │
   │   Autoregressive inference   │
   └──────────────────────────────┘
```

## Quick Start

### Prerequisites

```bash
# Python 3.10+
pip install torch datasets psutil
```

### 1. Build Vocabulary

```bash
python3 tokenizer.py
```

This downloads training data from HuggingFace and builds `vocab.json`. The tokenizer auto-detects the dataset schema (DPO pairs, messages, plain text, etc.).

### 2. Train the Model

```bash
python3 train.py
```

Training uses cosine LR scheduling with warmup, gradient accumulation, mixed-precision, and an auto-tuner that adjusts hyperparameters live. Metrics are written to `metrics.json` for the dashboard.

### 3. Launch the Dashboard

```bash
python3 dashboard.py
# → http://localhost:8080
```

The dashboard runs in a separate process so it never competes with GPU training.

### 4. Generate Text

```bash
python3 generate.py                          # interactive mode
python3 generate.py --prompt "Explain why"   # single prompt
```

## Tokenizer

The tokenizer uses a configurable word-level approach with an encoding registry inspired by [tiktoken](https://github.com/openai/tiktoken).

### TokenizerConfig

All tokenizer behavior is controlled by a single `TokenizerConfig` dataclass:

```python
from tokenizer import TokenizerConfig, WordTokenizer

config = TokenizerConfig(
    name="my-config",
    min_frequency=3,        # words appearing < 3 times → <UNK>
    max_vocab_size=50000,   # cap vocab at 50k tokens
    lowercase=True,
    keep_punctuation=[".", ",", "!", "?"],
    strip_accents=True,     # café → cafe
    unk_threshold=0.10,     # warn if >10% tokens are unknown
)

tok = WordTokenizer(config=config)
```

### Encoding Presets

Three built-in presets cover common use cases:

```python
from tokenizer import WordTokenizer

tok = WordTokenizer.from_encoding("default")         # balanced general-purpose
tok = WordTokenizer.from_encoding("conversational")   # chat/DPO data
tok = WordTokenizer.from_encoding("technical")        # code & math
```

| Preset | Min Freq | Lowercase | Extra Punctuation | Best For |
|--------|----------|-----------|-------------------|----------|
| `default` | 5 | ✓ | `.,!?;:` | General text |
| `conversational` | 3 | ✓ | `.,!?;:'"` | Chat / dialogue |
| `technical` | 2 | ✗ | `.,()[]{}=+-*/:;` | Code / math |

Register a custom encoding:

```python
from tokenizer import register_encoding, TokenizerConfig

register_encoding("my-preset", TokenizerConfig(
    min_frequency=2,
    lowercase=False,
    keep_punctuation=[".", ",", ":", ";", "→"],
))
```

### Encode & Decode

```python
tok = WordTokenizer.load("vocab.json")

ids = tok.encode("Attention is all you need", maxLength=512)
# → [2, 45, 12, 89, 7, 231, 3, 0, 0, ...]
#    SOS  ↑ words ↑              EOS  PAD...

text = tok.decode(ids)
# → "attention is all you need"
```

### Runtime Config Updates

```python
tok.update_config(min_frequency=2, lowercase=False)
tok.buildVocab(texts)  # rebuild with new settings
tok.save("vocab.json")
```

### Dashboard Settings

All tokenizer settings are editable from the **Token Visualizer** tab in the dashboard:

- **Encoding presets** — one-click switching between `default`, `conversational`, `technical`
- **Config fields** — min frequency, vocab cap, regex pattern, punctuation, toggles
- **Dataset controls** — source, HuggingFace dataset name, row count
- **Rebuild** — regenerate `vocab.json` from the dashboard with a confirmation modal

## Configuration

All settings live in [`config.py`](config.py) and are validated at startup:

```python
# config.py (excerpt)
MODEL = {
    "dModel": 256,       # embedding dimension
    "dFF": 1024,         # feed-forward hidden size
    "h": 8,              # attention heads
    "N": 4,              # encoder/decoder layers
    "dropout": 0.1,
    "seqLength": 512,
}

TRAINING = {
    "epochs": 50,
    "batchSize": 10,
    "maxLR": 5e-4,
    "warmupSteps": 500,
    "labelSmoothing": 0.2,
    "gradClipNorm": 1.1,
    "accumSteps": 4,       # effective batch = 10 × 4 = 40
}

TOKENIZER = {
    "minFrequency": 5,
    "vocabArticles": 52000,
    "encoding": "default",       # preset name
    "maxVocabSize": None,        # None = unlimited
    "lowercase": True,
    "unkThreshold": 0.15,
}
```

## File Reference

| File | Purpose |
|------|---------|
| `tokenizer.py` | Configurable word-level tokenizer with encoding registry |
| `model.py` | Transformer architecture (encoder-decoder, multi-head attention) |
| `train.py` | Training loop with cosine LR, gradient accumulation, auto-tuner |
| `generate.py` | Autoregressive text generation (greedy decoding) |
| `config.py` | Centralized configuration with validation |
| `dashboard.py` | HTTP server for the training dashboard |
| `dashboard.html` | Dashboard UI (charts, attention explorer, tokenizer settings) |
| `autotuner.py` | Live hyperparameter adjustment during training |
| `logging_config.py` | Structured logging with rotating file output |

## Dashboard

The dashboard provides real-time visibility into training:

- **Training tab** — loss, accuracy, perplexity, learning rate, gradient norms (live charts)
- **Config tab** — model vs. paper architecture comparison
- **Attention Explorer** — per-layer, per-head attention heatmaps
- **Token Visualizer** — encode text and see token IDs + tokenizer settings panel
- **Embedding Map** — 3D PCA projection of learned word embeddings
- **Settings** — theme switching

## License

This project is for educational and research purposes.
