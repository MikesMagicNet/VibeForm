# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  tokenizer.py  –  Configurable Word-Level Tokenizer                       ║
# ║                                                                            ║
# ║  Architecture inspired by tiktoken / HuggingFace Tokenizers:               ║
# ║    • TokenizerConfig  — single dataclass for ALL tokenizer settings        ║
# ║    • Encoding registry — named presets (default, conversational, technical)║
# ║    • WordTokenizer     — fully configurable via config or encoding name    ║
# ║                                                                            ║
# ║  HOW IT CONNECTS TO model.py:                                              ║
# ║    Raw text                                                                ║
# ║       ↓  tokenizer.encode()     ← THIS FILE                               ║
# ║    List of integer IDs                                                     ║
# ║       ↓  Inputs (nn.Embedding)  ← model.py                                ║
# ║    Dense vectors (dModel)                                                  ║
# ║       ↓  PositionalEncoding     ← model.py                                ║
# ║    Vectors + position info                                                 ║
# ║       ↓  Encoder / Decoder      ← model.py                                ║
# ║       ↓  ProjectionLayer        ← model.py                                ║
# ║    Probability over vocab                                                  ║
# ║       ↓  tokenizer.decode()     ← THIS FILE                               ║
# ║    Human-readable text                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import re
import json
import os
import time
import copy
import logging
import unicodedata
from dataclasses import dataclass, field, asdict
from typing import Optional
from collections import Counter
from datasets import load_dataset


# ═══════════════════════════════════════════════════════════════════════════════
#  TOKENIZER CONFIG  –  Single source of truth for tokenizer behavior
#
#  Every setting that controls how text is split, filtered, and mapped lives
#  here.  Pass a TokenizerConfig to WordTokenizer, or use a named preset
#  from the encoding registry.
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TokenizerConfig:
    """
    Complete configuration for the WordTokenizer.

    Attributes:
        name:              Human-readable name for this configuration.
        min_frequency:     Minimum word count to include in vocab (rare → <UNK>).
        max_vocab_size:    Optional hard cap on vocabulary size (None = unlimited).
        lowercase:         Lowercase all input text before tokenization.
        special_tokens:    Mapping of role → token string.
        token_pattern:     Regex pattern for initial text splitting.
        keep_punctuation:  Characters kept as individual tokens.
        strip_accents:     Normalize accented characters to ASCII.
        unk_threshold:     UNK rate above this triggers a warning (0.0–1.0).
        dataset_source:    Which loader to use ("claude-opus" or "wikipedia").
        dataset_name:      HuggingFace dataset identifier.
        dataset_rows:      Number of rows to scan when building vocabulary.
        seq_length:        Sequence length used during tokenizer testing.
    """
    name: str = "default"
    min_frequency: int = 5
    max_vocab_size: Optional[int] = None
    lowercase: bool = True
    special_tokens: dict = field(default_factory=lambda: {
        "pad": "<PAD>",
        "unk": "<UNK>",
        "sos": "<SOS>",
        "eos": "<EOS>",
    })
    token_pattern: str = r"([^\w\s\-])"
    keep_punctuation: list = field(default_factory=lambda: [".", ",", "!", "?", ";", ":"])
    strip_accents: bool = False
    unk_threshold: float = 0.15
    dataset_source: str = "claude-opus"
    dataset_name: str = "math-extraction-comp/deepseek-ai__deepseek-llm-67b-chat"
    dataset_rows: int = 52000
    seq_length: int = 512

    def to_dict(self):
        """Serialize to a plain dict (JSON-safe)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        """Reconstruct from a dict, ignoring unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def validate(self):
        """
        Validates config values and raises ValueError for invalid settings.
        Returns list of warning strings for non-fatal issues.
        """
        warnings = []

        if self.min_frequency <= 0:
            raise ValueError(f"min_frequency must be positive, got {self.min_frequency}")
        if self.max_vocab_size is not None and self.max_vocab_size <= 0:
            raise ValueError(f"max_vocab_size must be positive or None, got {self.max_vocab_size}")
        if not (0.0 <= self.unk_threshold <= 1.0):
            raise ValueError(f"unk_threshold must be in [0, 1], got {self.unk_threshold}")
        if self.seq_length <= 0:
            raise ValueError(f"seq_length must be positive, got {self.seq_length}")
        if self.dataset_rows <= 0:
            raise ValueError(f"dataset_rows must be positive, got {self.dataset_rows}")

        required_roles = {"pad", "unk", "sos", "eos"}
        missing = required_roles - set(self.special_tokens.keys())
        if missing:
            raise ValueError(f"special_tokens missing required roles: {missing}")

        if self.min_frequency > 20:
            warnings.append(
                f"min_frequency={self.min_frequency} is high — vocab will be small, "
                f"many words mapped to <UNK>."
            )
        if self.max_vocab_size is not None and self.max_vocab_size < 1000:
            warnings.append(
                f"max_vocab_size={self.max_vocab_size} is very small — "
                f"model capacity will be limited."
            )

        return warnings


# ── Legacy aliases (backward compat with train.py, generate.py, etc.) ─────────

_DEFAULT_CONFIG = TokenizerConfig()

PAD_TOKEN = _DEFAULT_CONFIG.special_tokens["pad"]
UNK_TOKEN = _DEFAULT_CONFIG.special_tokens["unk"]
SOS_TOKEN = _DEFAULT_CONFIG.special_tokens["sos"]
EOS_TOKEN = _DEFAULT_CONFIG.special_tokens["eos"]

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]


# ═══════════════════════════════════════════════════════════════════════════════
#  ENCODING REGISTRY  –  Named presets for common use cases
#
#  Usage:
#    config = get_encoding("conversational")
#    tokenizer = WordTokenizer(config=config)
#
#  Or register your own:
#    register_encoding("my-custom", TokenizerConfig(name="my-custom", ...))
# ═══════════════════════════════════════════════════════════════════════════════

_ENCODINGS: dict[str, TokenizerConfig] = {}


def register_encoding(name: str, config: TokenizerConfig):
    """Register a named encoding preset in the global registry."""
    config.name = name
    _ENCODINGS[name] = config


def get_encoding(name: str) -> TokenizerConfig:
    """Retrieve a registered encoding preset by name."""
    if name not in _ENCODINGS:
        available = ", ".join(_ENCODINGS.keys()) or "(none)"
        raise KeyError(f"Unknown encoding '{name}'. Available: {available}")
    return copy.deepcopy(_ENCODINGS[name])


def list_encodings() -> list[dict]:
    """Returns all registered encoding presets as a list of dicts."""
    return [
        {"name": name, "config": cfg.to_dict()}
        for name, cfg in _ENCODINGS.items()
    ]


# ── Built-in presets ──────────────────────────────────────────────────────────

register_encoding("default", TokenizerConfig(
    name="default",
    min_frequency=5,
    lowercase=True,
    token_pattern=r"([^\w\s\-])",
    keep_punctuation=[".", ",", "!", "?", ";", ":"],
    strip_accents=False,
    unk_threshold=0.15,
))

register_encoding("conversational", TokenizerConfig(
    name="conversational",
    min_frequency=3,
    lowercase=True,
    token_pattern=r"([^\w\s\-'])",
    keep_punctuation=[".", ",", "!", "?", ";", ":", "'", '"', "-"],
    strip_accents=False,
    unk_threshold=0.10,
))

register_encoding("technical", TokenizerConfig(
    name="technical",
    min_frequency=2,
    lowercase=False,
    token_pattern=r"([^\w\s\-_.])",
    keep_punctuation=[".", ",", "(", ")", "[", "]", "{", "}", "=", "+", "-", "*", "/", ":", ";"],
    strip_accents=True,
    unk_threshold=0.20,
))


# ═══════════════════════════════════════════════════════════════════════════════
#  TOKENIZATION  –  Splitting raw text into a list of word tokens
# ═══════════════════════════════════════════════════════════════════════════════

def tokenize(text, config=None):
    """
    Splits a raw string into tokens using the given TokenizerConfig.

    Falls back to the default config when called without arguments,
    preserving backward compatibility with the rest of the codebase.

    Args:
        text (str):                 Raw input string.
        config (TokenizerConfig):   Settings controlling tokenization behavior.

    Returns:
        list[str]: Word tokens.
    """
    if config is None:
        config = _DEFAULT_CONFIG

    # Optional accent stripping (e.g., café → cafe)
    if config.strip_accents:
        text = unicodedata.normalize("NFD", text)
        text = "".join(c for c in text if unicodedata.category(c) != "Mn")

    # Optional lowercasing
    if config.lowercase:
        text = text.lower()

    # Separate punctuation from words using the configured pattern
    text = re.sub(config.token_pattern, r" \1 ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Build the set of allowed punctuation for fast lookup
    allowed_punct = set(config.keep_punctuation)

    # Keep alphanumeric/underscore/hyphen tokens AND allowed punctuation
    tokens = [
        t for t in text.split()
        if re.match(r"^[\w\-]+$", t) or t in allowed_punct
    ]
    return tokens


# ═══════════════════════════════════════════════════════════════════════════════
#  WORD-LEVEL TOKENIZER CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class WordTokenizer:
    """
    A configurable word-level tokenizer that maps words ↔ integer IDs.

    Vocabulary is built from a training corpus.  Words that appear fewer than
    `config.min_frequency` times are excluded (they become <UNK> at encode time).

    Attributes:
        config (TokenizerConfig):  All tokenizer settings.
        vocabSize (int):   Total number of unique tokens (words + special tokens).
        word2id (dict):    Maps a word string → integer ID.
        id2word (dict):    Maps an integer ID → word string (reverse lookup).
    """

    # ─── CONSTRUCTION ─────────────────────────────────────────────────────

    def __init__(self, config=None, *, minFrequency=None):
        """
        Create a new WordTokenizer.

        Args:
            config (TokenizerConfig): Full configuration.  Takes priority.
            minFrequency (int):       Legacy shortcut — creates a default
                                      config with the given min_frequency.
                                      Ignored if `config` is provided.
        """
        if config is not None:
            self.config = copy.deepcopy(config)
        elif minFrequency is not None:
            self.config = TokenizerConfig(min_frequency=minFrequency)
        else:
            self.config = TokenizerConfig()

        # Backward-compat alias used by train.py / generate.py
        self.minFrequency = self.config.min_frequency

        # Resolve special tokens from config
        self._pad = self.config.special_tokens["pad"]
        self._unk = self.config.special_tokens["unk"]
        self._sos = self.config.special_tokens["sos"]
        self._eos = self.config.special_tokens["eos"]
        self._special_list = [self._pad, self._unk, self._sos, self._eos]

        # Initialize vocabulary with special tokens first (IDs 0–3)
        self.word2id = {}
        self.id2word = {}
        for i, token in enumerate(self._special_list):
            self.word2id[token] = i
            self.id2word[i] = token

        self.vocabSize = len(self._special_list)

        # Debug / diagnostic tracking
        self._encodeCount = 0
        self._totalTokensEncoded = 0
        self._totalUnkTokens = 0
        self._unkWords = Counter()
        self._encodeTimes = []

    @classmethod
    def from_encoding(cls, name: str):
        """
        Create a WordTokenizer from a named encoding preset.

        Args:
            name (str): Encoding name (e.g., "default", "conversational").

        Returns:
            WordTokenizer: A new tokenizer with that preset's config.

        Example:
            >>> tok = WordTokenizer.from_encoding("conversational")
        """
        config = get_encoding(name)
        return cls(config=config)

    # ─── CONFIG ACCESS ────────────────────────────────────────────────────

    def get_config(self) -> dict:
        """Returns the current config as a JSON-serializable dict."""
        return self.config.to_dict()

    def update_config(self, **kwargs):
        """
        Update individual config fields at runtime.

        Args:
            **kwargs: Field names and their new values.

        Example:
            >>> tok.update_config(min_frequency=3, lowercase=False)
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise KeyError(f"Unknown config field: '{key}'")

        # Keep legacy alias in sync
        self.minFrequency = self.config.min_frequency

        # Re-resolve special tokens if they changed
        self._pad = self.config.special_tokens.get("pad", "<PAD>")
        self._unk = self.config.special_tokens.get("unk", "<UNK>")
        self._sos = self.config.special_tokens.get("sos", "<SOS>")
        self._eos = self.config.special_tokens.get("eos", "<EOS>")
        self._special_list = [self._pad, self._unk, self._sos, self._eos]

    # ─── BUILD VOCABULARY ─────────────────────────────────────────────────

    def buildVocab(self, texts):
        """
        Scans a list of raw text strings to discover all words and their
        frequencies, then creates the word ↔ ID mappings.

        How it works:
          1. Tokenize every text into words (using self.config).
          2. Count how often each word appears.
          3. Keep only words that appear >= min_frequency times.
          4. Optionally cap at max_vocab_size (most frequent words win).
          5. Assign each surviving word a unique integer ID.

        Args:
            texts (list[str]): A list of raw strings (e.g., articles, prompts).

        After calling this:
            - self.word2id  is fully populated
            - self.id2word  is fully populated
            - self.vocabSize reflects the final vocabulary size
        """
        print("Building vocabulary...")

        # Step 1 & 2: Count every word across all texts
        wordCounts = Counter()
        for text in texts:
            words = tokenize(text, config=self.config)
            wordCounts.update(words)

        # Step 3: Filter by frequency
        candidates = {
            word: count
            for word, count in wordCounts.items()
            if count >= self.config.min_frequency
        }

        # Step 4: Optional vocab cap (keep most frequent)
        if self.config.max_vocab_size is not None:
            max_words = self.config.max_vocab_size - len(self._special_list)
            if len(candidates) > max_words:
                candidates = dict(
                    sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:max_words]
                )

        # Step 5: Assign IDs
        # Reset to special tokens only
        self.word2id = {}
        self.id2word = {}
        for i, token in enumerate(self._special_list):
            self.word2id[token] = i
            self.id2word[i] = token

        nextId = len(self._special_list)
        for word in candidates:
            self.word2id[word] = nextId
            self.id2word[nextId] = word
            nextId += 1

        self.vocabSize = nextId

        keptCount = len(candidates)
        droppedCount = len(wordCounts) - keptCount

        print(f"  Words seen:    {len(wordCounts):,}")
        print(f"  Words kept:    {keptCount:,}  (appeared >= {self.config.min_frequency} times)")
        print(f"  Words dropped: {droppedCount:,}  (too rare → mapped to <UNK>)")
        if self.config.max_vocab_size is not None:
            print(f"  Vocab cap:     {self.config.max_vocab_size:,}")
        print(f"  Final vocabSize: {self.vocabSize:,}")
        print(f"  (This number goes into Inputs(dModel, vocabSize) in model.py)\n")


    # ─── ENCODE: text → list of integer IDs ───────────────────────────────

    def encode(self, text, maxLength=None, addSpecialTokens=True):
        """
        Converts a raw text string into a list of integer IDs that the
        Transformer can process.

        Steps:
          1. Tokenize the text into words.
          2. Optionally prepend <SOS> and append <EOS>.
          3. Look up each word in word2id (unknown words → <UNK> ID).
          4. Optionally truncate or pad to a fixed maxLength.

        Args:
            text (str):              Raw input string.
            maxLength (int or None): If set, sequences are truncated or padded
                                     to exactly this length.
            addSpecialTokens (bool): If True, wraps the sequence with
                                     <SOS> ... <EOS>.

        Returns:
            list[int]: Integer IDs ready to become a torch.Tensor.

        Example:
            >>> tok.encode("The cat sat")
            [2, 4, 127, 853, 3]       # [<SOS>, the, cat, sat, <EOS>]

            >>> tok.encode("The cat sat", maxLength=8)
            [2, 4, 127, 853, 3, 0, 0, 0]  # padded with <PAD> to length 8
        """
        t0 = time.perf_counter()
        words = tokenize(text, config=self.config)

        # Look up each word → its integer ID (or <UNK>'s ID if not in vocab)
        unkId = self.word2id[self._unk]
        ids = []
        unkCount = 0
        for word in words:
            wid = self.word2id.get(word, unkId)
            if wid == unkId and word not in self._special_list:
                unkCount += 1
                self._unkWords[word] += 1
            ids.append(wid)

        # Track UNK rate for bottleneck detection
        self._encodeCount += 1
        self._totalTokensEncoded += len(words)
        self._totalUnkTokens += unkCount
        if words:
            unkRate = unkCount / len(words)
            if unkRate > self.config.unk_threshold:
                logging.warning(
                    f"Tokenizer: high UNK rate {unkRate:.1%} "
                    f"({unkCount}/{len(words)} tokens). "
                    f"Vocab may be too small or min_frequency too high."
                )

        # Wrap with <SOS> at the start and <EOS> at the end
        if addSpecialTokens:
            ids = [self.word2id[self._sos]] + ids + [self.word2id[self._eos]]

        # Truncate if too long
        if maxLength is not None:
            ids = ids[:maxLength]

        # Pad if too short
        if maxLength is not None:
            padId = self.word2id[self._pad]
            paddingNeeded = maxLength - len(ids)
            ids = ids + [padId] * paddingNeeded

        self._encodeTimes.append(time.perf_counter() - t0)
        return ids


    # ─── DECODE: list of integer IDs → text ───────────────────────────────

    def decode(self, ids, skipSpecialTokens=True):
        """
        Converts a list of integer IDs back into a human-readable string.
        This is the reverse of encode().

        Args:
            ids (list[int]):           Token IDs from the model.
            skipSpecialTokens (bool):  If True, removes <PAD>, <SOS>, <EOS>, <UNK>.

        Returns:
            str: The decoded text.

        Example:
            >>> tok.decode([2, 4, 127, 853, 3, 0, 0])
            "the cat sat"
        """
        words = []
        specialIds = set(range(len(self._special_list)))

        for tokenId in ids:
            if skipSpecialTokens and tokenId in specialIds:
                continue
            word = self.id2word.get(tokenId, self._unk)
            words.append(word)

        return " ".join(words)


    # ─── SELF-CHECK: round-trip validation ─────────────────────────────────

    def selfCheck(self, testSentences=None):
        """
        Validates tokenizer integrity by encoding then decoding test sentences
        and checking for information loss. Returns True if all checks pass.
        """
        if testSentences is None:
            testSentences = [
                "the transformer model uses attention",
                "deep learning is changing the world",
                "neural networks can learn patterns",
            ]

        passed = True
        for sentence in testSentences:
            encoded = self.encode(sentence, addSpecialTokens=True)
            decoded = self.decode(encoded, skipSpecialTokens=True)
            normalized = tokenize(sentence, config=self.config)
            decodedTokens = decoded.split()

            for word in normalized:
                if word in self.word2id and word not in decodedTokens:
                    logging.error(
                        f"Self-check FAIL: '{word}' is in vocab but "
                        f"was lost in round-trip for: '{sentence}'"
                    )
                    passed = False

        if passed:
            print("  ✅ Tokenizer self-check passed (encode↔decode round-trip OK)")
        else:
            print("  ⚠️  Tokenizer self-check found issues (see warnings above)")
        return passed


    # ─── DIAGNOSE: surface bottlenecks ─────────────────────────────────────

    def diagnose(self):
        """
        Prints a diagnostic report about tokenizer health.
        Call after processing data to see if the vocab is a bottleneck.
        """
        print("\n" + "─" * 50)
        print("  TOKENIZER DIAGNOSTICS")
        print("─" * 50)

        # Config summary
        print(f"  Encoding:        {self.config.name}")
        print(f"  Min frequency:   {self.config.min_frequency}")
        print(f"  Lowercase:       {self.config.lowercase}")
        cap = self.config.max_vocab_size or "unlimited"
        print(f"  Vocab cap:       {cap}")

        # Vocab stats
        specialCount = len(self._special_list)
        realWords = self.vocabSize - specialCount
        print(f"  Vocab size:      {self.vocabSize:,} ({realWords:,} words + {specialCount} special)")

        # UNK rate
        if self._totalTokensEncoded > 0:
            globalUnkRate = self._totalUnkTokens / self._totalTokensEncoded
            print(f"  Global UNK rate: {globalUnkRate:.2%} ({self._totalUnkTokens:,}/{self._totalTokensEncoded:,})")
            if globalUnkRate > 0.10:
                print(f"  ⚠️  UNK rate above 10% — consider lowering min_frequency")
            elif globalUnkRate > 0.05:
                print(f"  ⚡ UNK rate moderate — model may struggle with rare words")
            else:
                print(f"  ✅ UNK rate healthy")

            if self._unkWords:
                top = self._unkWords.most_common(10)
                print(f"  Top unknown words: {', '.join(w for w, _ in top)}")
        else:
            print(f"  (no encoding stats yet — call encode() first)")

        # Encoding speed
        if self._encodeTimes:
            avgMs = (sum(self._encodeTimes) / len(self._encodeTimes)) * 1000
            print(f"  Avg encode time: {avgMs:.3f} ms/call ({len(self._encodeTimes):,} calls)")
            if avgMs > 5.0:
                print(f"  ⚠️  Encoding is slow — consider caching or batch tokenization")
            else:
                print(f"  ✅ Encoding speed OK")

        print("─" * 50 + "\n")


    # ─── SAVE / LOAD  –  persist vocabulary + config to disk ──────────────

    def save(self, path):
        """
        Saves the vocabulary and config to a JSON file.

        The file stores word2id, id2word, vocabSize, and the full
        TokenizerConfig so settings are preserved across sessions.

        Args:
            path (str): File path, e.g. "vocab.json"
        """
        data = {
            "config": self.config.to_dict(),
            "minFrequency": self.config.min_frequency,  # legacy compat
            "vocabSize": self.vocabSize,
            "word2id": self.word2id,
            "id2word": {str(k): v for k, v in self.id2word.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"Vocabulary saved to {path}  ({self.vocabSize:,} tokens)")


    @classmethod
    def load(cls, path):
        """
        Loads a previously saved vocabulary and config from a JSON file.

        Handles both new-format (with config) and legacy-format (without)
        vocab files for backward compatibility.

        Args:
            path (str): File path to the saved vocab, e.g. "vocab.json"

        Returns:
            WordTokenizer: A fully initialized tokenizer ready to encode/decode.
        """
        with open(path, "r") as f:
            data = json.load(f)

        # Reconstruct config (new format) or fall back to defaults (legacy)
        if "config" in data:
            config = TokenizerConfig.from_dict(data["config"])
        else:
            config = TokenizerConfig(min_frequency=data.get("minFrequency", 5))

        tokenizer = cls(config=config)
        tokenizer.word2id = data["word2id"]
        tokenizer.id2word = {int(k): v for k, v in data["id2word"].items()}
        tokenizer.vocabSize = data["vocabSize"]

        print(f"Vocabulary loaded from {path}  ({tokenizer.vocabSize:,} tokens)")
        return tokenizer


# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET LOADING  –  Fetch training data from HuggingFace
# ═══════════════════════════════════════════════════════════════════════════════

def loadDataOne(language="en", date="20231101", numArticles=50000):
    """
    Downloads Wikipedia articles from HuggingFace and returns them as a
    list of raw text strings.

    The dataset: https://huggingface.co/datasets/wikimedia/wikipedia
      - Each row has: id, url, title, text
      - We only need the 'text' field (the article body)

    Args:
        language (str):     Wikipedia language code.
                            "en" = English, "fr" = French, "es" = Spanish, etc.
        date (str):         Wikipedia dump date. "20231101" is the latest.
        numArticles (int):  How many articles to load. The full English Wikipedia
                            has ~6.4 million articles — start small for testing!
                            50,000 is a good starting point.

    Returns:
        list[str]: Raw article texts.
    """
    configName = f"{date}.{language}"
    print(f"Loading Wikipedia dataset: wikimedia/wikipedia [{configName}]")
    print(f"  Requesting {numArticles:,} articles...")

    dataset = load_dataset(
        "wikimedia/wikipedia",
        configName,
        split="train",
        streaming=True
    )

    texts = []
    for i, article in enumerate(dataset):
        if i >= numArticles:
            break
        texts.append(article["text"])

        if (i + 1) % 10000 == 0:
            print(f"  Loaded {i + 1:,} articles...")

    print(f"  Done! Loaded {len(texts):,} articles.\n")
    return texts


def loadDatasetTwo(datasetName="Roman1111111/claude-opus-4.6-10000x", numRows=9633):
    """
    Loads a dataset from HuggingFace, supporting multiple formats including DPO.

    For DPO datasets (prompt/chosen/rejected):
      Returns list of (prompt, response) tuples for encoder→decoder training.

    For messages or plain text datasets:
      Returns list of strings (flat text).

    Args:
        datasetName (str): HuggingFace dataset identifier.
        numRows (int):     How many rows to load.

    Returns:
        list[tuple[str,str]] or list[str]: Training data.
    """
    print(f"Loading dataset: {datasetName}")
    print(f"  Requesting {numRows:,} rows...")

    try:
        dataset = load_dataset(datasetName, split="train", streaming=True)
    except (RuntimeError, ConnectionError, OSError) as e:
        print(f"\n  ❌ Failed to connect to HuggingFace Hub: {type(e).__name__}")
        print(f"     Check your internet connection and try again.")
        print(f"     Dataset: {datasetName}")
        raise SystemExit(1)

    texts = []
    schema_type = None

    for i, row in enumerate(dataset):
        if i >= numRows:
            break

        if schema_type is None:
            if "messages" in row:
                schema_type = "messages"
            elif "instruction" in row and "output" in row:
                schema_type = "alpaca"
            elif "chosen" in row or "prompt" in row:
                schema_type = "dpo"
            elif "question" in row and ("answer" in row or "prediction" in row or "gold" in row or "extracted_answer" in row):
                schema_type = "math"
            elif "text" in row:
                schema_type = "text"
            else:
                schema_type = "fallback"
            print(f"  Detected schema: {schema_type}")

        if schema_type == "messages":
            parts = []
            for msg in row.get("messages", []):
                content = msg.get("content", "")
                reasoning = msg.get("reasoning", "")
                if content:
                    parts.append(content)
                if reasoning:
                    parts.append(reasoning)
            if parts:
                texts.append(" ".join(parts))

        elif schema_type == "alpaca":
            instruction = row.get("instruction", "")
            input_text = row.get("input", "")
            prompt = f"{instruction} {input_text}".strip() if input_text else instruction
            output = row.get("output", "")
            if prompt and output:
                texts.append((prompt, output))

        elif schema_type == "dpo":
            prompt = row.get("prompt", "")
            if isinstance(prompt, list):
                prompt = " ".join(
                    m.get("content", "") for m in prompt if isinstance(m, dict)
                )
            chosen = row.get("chosen", "")
            if isinstance(chosen, list):
                chosen = " ".join(
                    m.get("content", "") for m in chosen if isinstance(m, dict)
                )
            if prompt and chosen:
                texts.append((prompt, chosen))

        elif schema_type == "math":
            prompt = str(row.get("question", ""))
            output = str(row.get("prediction", "") or row.get("answer", "") or row.get("extracted_answer", "") or row.get("gold", ""))
            if prompt and output:
                texts.append((prompt, output))

        elif schema_type == "text":
            text = row.get("text", "")
            if text:
                texts.append(text)

        else: # fallback
            parts = []
            for key, val in row.items():
                if isinstance(val, str):
                    parts.append(val)
                elif isinstance(val, list):
                    for item in val:
                        if isinstance(item, dict):
                            if "content" in item:
                                parts.append(str(item["content"]))
                            elif "value" in item:
                                parts.append(str(item["value"]))
            if parts:
                texts.append(" ".join(parts))

        if (i + 1) % 2000 == 0:
            print(f"  Loaded {i + 1:,} rows...")

    print(f"  Done! Loaded {len(texts):,} rows.\n")
    return texts


def extractTextsForVocab(data):
    """
    Flattens training data (either list of strings or list of tuples) into
    a flat list of strings suitable for vocabulary building.

    Args:
        data: Output from loadClaudeOpusDataset — either list[str] or list[tuple[str,str]].

    Returns:
        list[str]: Flat list of text strings.
    """
    flat = []
    for item in data:
        if isinstance(item, (list, tuple)):
            flat.extend(item)  # unpack (prompt, response)
        else:
            flat.append(item)
    return flat


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN  –  Build the tokenizer from training data and save it
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    from config import TOKENIZER, PATHS, DATA

    # ── Configuration pulled from config.py ───────────────────────────────
    VOCAB_PATH   = PATHS["vocab"]
    NUM_ARTICLES = TOKENIZER["vocabArticles"]
    MIN_FREQ     = TOKENIZER["minFrequency"]
    SEQ_LENGTH   = TOKENIZER["seqLengthForTest"]

    # Build a TokenizerConfig from config.py settings
    tokConfig = TokenizerConfig(
        name=TOKENIZER.get("encoding", "default"),
        min_frequency=MIN_FREQ,
        max_vocab_size=TOKENIZER.get("maxVocabSize", None),
        lowercase=TOKENIZER.get("lowercase", True),
        strip_accents=TOKENIZER.get("stripAccents", False),
        unk_threshold=TOKENIZER.get("unkThreshold", 0.15),
        dataset_source=DATA["source"],
        dataset_name=DATA.get("claudeDataset", ""),
        dataset_rows=NUM_ARTICLES,
        seq_length=SEQ_LENGTH,
    )

    # ── Step 1: Load data ─────────────────────────────────────────────────
    print("=" * 60)
    print(f"  STEP 1: Loading Data ({DATA['source']})")
    print("=" * 60)

    if os.path.exists(VOCAB_PATH):
        print(f"  Found existing vocab at '{VOCAB_PATH}', loading it instead.\n")
        tokenizer = WordTokenizer.load(VOCAB_PATH)
    else:
        if DATA["source"] == "claude-opus":
            texts = loadDatasetTwo(
                datasetName=DATA["claudeDataset"],
                numRows=NUM_ARTICLES,
            )
        else:
            texts = loadDataOne(numArticles=NUM_ARTICLES)

        # ── Step 2: Build vocabulary ──────────────────────────────────────
        print("=" * 60)
        print("  STEP 2: Building Vocabulary")
        print("=" * 60)

        tokenizer = WordTokenizer(config=tokConfig)
        tokenizer.buildVocab(extractTextsForVocab(texts))

        # ── Step 3: Save vocabulary ───────────────────────────────────────
        print("=" * 60)
        print("  STEP 3: Saving Vocabulary")
        print("=" * 60)

        tokenizer.save(VOCAB_PATH)

    # ── Step 4: Test encode / decode ──────────────────────────────────────
    print("=" * 60)
    print("  STEP 4: Testing Encode / Decode")
    print("=" * 60)

    testSentence = "The Transformer model was introduced in the paper Attention Is All You Need"
    print(f"\n  Input:    \"{testSentence}\"")

    encoded = tokenizer.encode(testSentence, maxLength=SEQ_LENGTH)
    print(f"  Encoded:  {encoded[:20]}...  (showing first 20 of {len(encoded)} IDs)")

    decoded = tokenizer.decode(encoded)
    print(f"  Decoded:  \"{decoded}\"")

    # Show vocab size
    print(f"\n  ┌─────────────────────────────────────────────┐")
    print(f"  │  vocabSize = {tokenizer.vocabSize:<10,}                    │")
    print(f"  │  Use this value in buildTransformer():       │")
    print(f"  │    source_vocabSize = {tokenizer.vocabSize:<10,}           │")
    print(f"  │    target_vocabSize = {tokenizer.vocabSize:<10,}           │")
    print(f"  └─────────────────────────────────────────────┘")
    print()
