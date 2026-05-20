# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  tokenizer.py  –  Word-Level Tokenizer for our Transformer                ║
# ║                                                                            ║
# ║  WHAT IS A TOKENIZER?                                                      ║
# ║  A tokenizer converts raw text (strings) into numbers (integers) that a    ║
# ║  neural network can process.  Our Transformer's Inputs layer               ║
# ║  (nn.Embedding) expects integer IDs, not words — so the tokenizer is the   ║
# ║  bridge between human language and the model.                              ║
# ║                                                                            ║
# ║  WHY "WORD-LEVEL"?                                                         ║
# ║  Each unique word in the training data gets its own integer ID.            ║
# ║  Example:  "the cat sat" → [4, 127, 853]                                  ║
# ║                                                                            ║
# ║  HOW IT CONNECTS TO model.py:                                              ║
# ║    Raw text                                                                ║
# ║       ↓  tokenizer.encode()     ← THIS FILE                               ║
# ║    List of integer IDs                                                     ║
# ║       ↓  Inputs (nn.Embedding)  ← model.py line 5                         ║
# ║    Dense vectors (dModel=512)                                              ║
# ║       ↓  PositionalEncoding     ← model.py line 15                        ║
# ║    Vectors + position info                                                 ║
# ║       ↓  Encoder / Decoder      ← model.py lines 120-169                  ║
# ║       ↓  ProjectionLayer        ← model.py line 171                       ║
# ║    Probability over vocab                                                  ║
# ║       ↓  tokenizer.decode()     ← THIS FILE                               ║
# ║    Human-readable text                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import re
import json
import os
import time
import logging
from collections import Counter
from datasets import load_dataset


# ── Special Tokens ────────────────────────────────────────────────────────────
# These are reserved words that the model uses for structure, not meaning.
#
#   <PAD>  – padding; fills empty slots so every sequence is the same length.
#            The Transformer's mask will ignore these positions.
#
#   <UNK>  – unknown; replaces any word not seen during training.
#
#   <SOS>  – start-of-sequence; tells the decoder "begin generating here."
#            Fed as the very first token to the decoder input.
#
#   <EOS>  – end-of-sequence; tells the decoder "stop generating."
#            The model learns to output this when the sentence is done.

PAD_TOKEN = "<PAD>"   # ... ID will always be 0
UNK_TOKEN = "<UNK>"   # ... ID will always be 1
SOS_TOKEN = "<SOS>"   # ... ID will always be 2
EOS_TOKEN = "<EOS>"   # ... ID will always be 3

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]


# ═══════════════════════════════════════════════════════════════════════════════
#  TOKENIZATION HELPER  – splitting raw text into a list of words
#
#  Aligned for DPO conversational data (tungdqzenai/grok_humanlike_dpo):
#    - Preserves underscores, hyphens (common in code + reasoning)
#    - Keeps common punctuation (.,!?;:) as individual tokens
#    - Keeps numbers intact
#    - Handles prompt/chosen/rejected conversational text
# ═══════════════════════════════════════════════════════════════════════════════

def tokenize(text):
    """
    Splits a raw string into lowercase tokens, tuned for DPO conversational
    data where punctuation carries meaning in dialogue.

    Keeps: letters, digits, underscores, hyphens (within words)
    Keeps: common punctuation as standalone tokens (.,!?;:-)
    Drops: rare symbols that add noise

    Args:
        text (str): Raw input string.

    Returns:
        list[str]: Word tokens.
    """
    text = text.lower()
    # Separate punctuation from words but keep word-internal hyphens/underscores
    text = re.sub(r"([^\w\s\-])", r" \1 ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Keep alphanumeric + underscore + hyphen tokens, AND common punctuation
    tokens = [t for t in text.split() if re.match(r"^[\w\-]+$", t) or t in ".,!?;:"]
    return tokens


# ═══════════════════════════════════════════════════════════════════════════════
#  WORD-LEVEL TOKENIZER CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class WordTokenizer:
    """
    A word-level tokenizer that maps words ↔ integer IDs.

    Vocabulary is built from a training corpus. Words that appear fewer than
    `minFrequency` times are excluded (they become <UNK> at encode time).

    Attributes:
        vocabSize (int):  Total number of unique tokens (words + special tokens).
                          This value is passed to model.py's:
                            - Inputs(dModel, vocabSize)        → embedding table
                            - ProjectionLayer(dModel, vocabSize) → output layer

        word2id (dict):   Maps a word string → integer ID.
                          Example: {"<PAD>": 0, "the": 4, "cat": 127, ...}

        id2word (dict):   Maps an integer ID → word string  (reverse lookup).
                          Example: {0: "<PAD>", 4: "the", 127: "cat", ...}
    """

    def __init__(self, minFrequency=5):
        """
        Args:
            minFrequency (int): Minimum number of times a word must appear in
                                the training data to earn its own ID. Rare words
                                below this count are mapped to <UNK>.
                                Higher = smaller vocab (faster, less memory).
                                Lower  = bigger vocab  (captures rare words).
        """
        self.minFrequency = minFrequency  # ... threshold for keeping a word

        # --- Initialize vocabulary with special tokens first ---
        # Special tokens always occupy the first IDs: 0, 1, 2, 3
        self.word2id = {}    # ... word string → integer ID
        self.id2word = {}    # ... integer ID → word string
        for i, token in enumerate(SPECIAL_TOKENS):
            self.word2id[token] = i
            self.id2word[i] = token

        self.vocabSize = len(SPECIAL_TOKENS)  # ... starts at 4, grows as we add words

        # Debug tracking
        self._encodeCount = 0
        self._totalTokensEncoded = 0
        self._totalUnkTokens = 0
        self._unkWords = Counter()  # tracks which words hit UNK most often
        self._encodeTimes = []


    # ─── BUILD VOCABULARY ─────────────────────────────────────────────────────

    def buildVocab(self, texts):
        """
        Scans a list of raw text strings to discover all words and their
        frequencies, then creates the word ↔ ID mappings.

        How it works:
          1. Tokenize every text into words.
          2. Count how often each word appears (Counter).
          3. Keep only words that appear >= minFrequency times.
          4. Assign each surviving word a unique integer ID.

        Args:
            texts (list[str]): A list of raw strings (e.g., Wikipedia articles).

        After calling this:
            - self.word2id  is fully populated
            - self.id2word  is fully populated
            - self.vocabSize reflects the final vocabulary size
        """
        print("Building vocabulary...")

        # Step 1 & 2: Count every word across all texts
        wordCounts = Counter()  # ... dict-like: {"the": 98234, "cat": 412, ...}
        for text in texts:
            words = tokenize(text)
            wordCounts.update(words)  # ... adds each word's count

        # Step 3 & 4: Filter by frequency, assign IDs
        # nextId starts after special tokens (which already took IDs 0–3)
        nextId = len(SPECIAL_TOKENS)  # ... = 4
        keptCount = 0
        droppedCount = 0

        for word, count in wordCounts.items():
            if count >= self.minFrequency:
                # This word appears often enough → give it an ID
                self.word2id[word] = nextId
                self.id2word[nextId] = word
                nextId += 1
                keptCount += 1
            else:
                # Too rare → will map to <UNK> at encode time
                droppedCount += 1

        self.vocabSize = nextId  # ... total tokens = special + kept words

        print(f"  Words seen:    {len(wordCounts):,}")
        print(f"  Words kept:    {keptCount:,}  (appeared >= {self.minFrequency} times)")
        print(f"  Words dropped: {droppedCount:,}  (too rare → mapped to <UNK>)")
        print(f"  Final vocabSize: {self.vocabSize:,}")
        print(f"  (This number goes into Inputs(dModel, vocabSize) in model.py)\n")


    # ─── ENCODE: text → list of integer IDs ───────────────────────────────────

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
                                     to exactly this length. This must match the
                                     sequenceLength used in PositionalEncoding
                                     (model.py line 17).
            addSpecialTokens (bool): If True, wraps the sequence with
                                     <SOS> ... <EOS>.

        Returns:
            list[int]: Integer IDs ready to become a torch.Tensor and be fed
                       into the model.

        Example:
            >>> tok.encode("The cat sat")
            [2, 4, 127, 853, 3]       # [<SOS>, the, cat, sat, <EOS>]

            >>> tok.encode("The cat sat", maxLength=8)
            [2, 4, 127, 853, 3, 0, 0, 0]  # padded with <PAD> to length 8
        """
        t0 = time.perf_counter()
        words = tokenize(text)

        # Look up each word → its integer ID (or <UNK>'s ID if not in vocab)
        unkId = self.word2id[UNK_TOKEN]  # ... = 1
        ids = []
        unkCount = 0
        for word in words:
            wid = self.word2id.get(word, unkId)
            if wid == unkId and word not in SPECIAL_TOKENS:
                unkCount += 1
                self._unkWords[word] += 1
            ids.append(wid)

        # Track UNK rate for bottleneck detection
        self._encodeCount += 1
        self._totalTokensEncoded += len(words)
        self._totalUnkTokens += unkCount
        if words:
            unkRate = unkCount / len(words)
            if unkRate > 0.15:
                logging.warning(
                    f"Tokenizer: high UNK rate {unkRate:.1%} "
                    f"({unkCount}/{len(words)} tokens). "
                    f"Vocab may be too small or minFrequency too high."
                )

        # Wrap with <SOS> at the start and <EOS> at the end
        if addSpecialTokens:
            ids = [self.word2id[SOS_TOKEN]] + ids + [self.word2id[EOS_TOKEN]]

        # Truncate if too long (must fit within model's sequenceLength)
        if maxLength is not None:
            ids = ids[:maxLength]

        # Pad if too short (fill remaining slots with <PAD> = 0)
        if maxLength is not None:
            padId = self.word2id[PAD_TOKEN]
            paddingNeeded = maxLength - len(ids)
            ids = ids + [padId] * paddingNeeded

        self._encodeTimes.append(time.perf_counter() - t0)
        return ids


    # ─── DECODE: list of integer IDs → text ───────────────────────────────────

    def decode(self, ids, skipSpecialTokens=True):
        """
        Converts a list of integer IDs back into a human-readable string.
        This is the reverse of encode() — used to read the model's output.

        Args:
            ids (list[int]):           Token IDs from the model.
            skipSpecialTokens (bool):  If True, removes <PAD>, <SOS>, <EOS>, <UNK>
                                       from the output.

        Returns:
            str: The decoded text.

        Example:
            >>> tok.decode([2, 4, 127, 853, 3, 0, 0])
            "the cat sat"
        """
        words = []
        specialIds = set(range(len(SPECIAL_TOKENS)))  # ... {0, 1, 2, 3}

        for tokenId in ids:
            if skipSpecialTokens and tokenId in specialIds:
                continue  # ... skip <PAD>, <UNK>, <SOS>, <EOS>
            word = self.id2word.get(tokenId, UNK_TOKEN)
            # ... look up the integer → word, fallback to "<UNK>"
            words.append(word)

        return " ".join(words)  # ... glue words back together with spaces


    # ─── SELF-CHECK: round-trip validation ─────────────────────────────────────

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
            normalized = tokenize(sentence)
            decodedTokens = decoded.split()

            # Check each non-UNK word survived the round trip
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


    # ─── DIAGNOSE: surface bottlenecks ─────────────────────────────────────────

    def diagnose(self):
        """
        Prints a diagnostic report about tokenizer health.
        Call after processing data to see if the vocab is a bottleneck.
        """
        print("\n" + "─" * 50)
        print("  TOKENIZER DIAGNOSTICS")
        print("─" * 50)

        # Vocab stats
        specialCount = len(SPECIAL_TOKENS)
        realWords = self.vocabSize - specialCount
        print(f"  Vocab size:      {self.vocabSize:,} ({realWords:,} words + {specialCount} special)")

        # UNK rate (the #1 tokenizer bottleneck)
        if self._totalTokensEncoded > 0:
            globalUnkRate = self._totalUnkTokens / self._totalTokensEncoded
            print(f"  Global UNK rate: {globalUnkRate:.2%} ({self._totalUnkTokens:,}/{self._totalTokensEncoded:,})")
            if globalUnkRate > 0.10:
                print(f"  ⚠️  UNK rate above 10% — consider lowering minFrequency")
            elif globalUnkRate > 0.05:
                print(f"  ⚡ UNK rate moderate — model may struggle with rare words")
            else:
                print(f"  ✅ UNK rate healthy")

            # Top unknown words
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


    # ─── SAVE / LOAD  –  persist the vocabulary to disk ───────────────────────

    def save(self, path):
        """
        Saves the vocabulary to a JSON file so you don't have to rebuild
        it every time. The file stores word2id, id2word, and minFrequency.

        Args:
            path (str): File path, e.g. "vocab.json"
        """
        data = {
            "minFrequency": self.minFrequency,
            "vocabSize": self.vocabSize,
            "word2id": self.word2id,
            # ... id2word keys must be strings in JSON, so we convert
            "id2word": {str(k): v for k, v in self.id2word.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"Vocabulary saved to {path}  ({self.vocabSize:,} tokens)")


    @classmethod
    def load(cls, path):
        """
        Loads a previously saved vocabulary from a JSON file.

        Args:
            path (str): File path to the saved vocab, e.g. "vocab.json"

        Returns:
            WordTokenizer: A fully initialized tokenizer ready to encode/decode.
        """
        with open(path, "r") as f:
            data = json.load(f)

        tokenizer = cls(minFrequency=data["minFrequency"])
        tokenizer.word2id = data["word2id"]
        # ... JSON keys are always strings, so convert back to int
        tokenizer.id2word = {int(k): v for k, v in data["id2word"].items()}
        tokenizer.vocabSize = data["vocabSize"]

        print(f"Vocabulary loaded from {path}  ({tokenizer.vocabSize:,} tokens)")
        return tokenizer


# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET LOADING  –  Fetch Wikipedia articles from HuggingFace
# ═══════════════════════════════════════════════════════════════════════════════

def loadWikipediaDataset(language="en", date="20231101", numArticles=50000):
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
    # The config name combines the date and language: "20231101.en"
    configName = f"{date}.{language}"
    print(f"Loading Wikipedia dataset: wikimedia/wikipedia [{configName}]")
    print(f"  Requesting {numArticles:,} articles...")

    # load_dataset downloads & caches the data from HuggingFace
    # streaming=True means we don't download the entire 71.8 GB at once —
    # instead it streams articles one at a time (much friendlier on RAM/disk)
    dataset = load_dataset(
        "wikimedia/wikipedia",
        configName,
        split="train",
        streaming=True       # ... stream instead of downloading everything
    )

    # Collect the article text from each row
    texts = []
    for i, article in enumerate(dataset):
        if i >= numArticles:
            break  # ... we have enough articles
        texts.append(article["text"])  # ... each article is a dict with 'text' key

        # Print progress every 10,000 articles
        if (i + 1) % 10000 == 0:
            print(f"  Loaded {i + 1:,} articles...")

    print(f"  Done! Loaded {len(texts):,} articles.\n")
    return texts


def loadClaudeOpusDataset(datasetName="Roman1111111/claude-opus-4.6-10000x", numRows=9633):
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
            # Extract prompt
            prompt = row.get("prompt", "")
            if isinstance(prompt, list):
                prompt = " ".join(
                    m.get("content", "") for m in prompt if isinstance(m, dict)
                )
            # Extract chosen response (the preferred answer)
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
#  MAIN  –  Build the tokenizer from Wikipedia and save it
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    from config import TOKENIZER, PATHS, DATA

    # ── Configuration pulled from config.py ───────────────────────────────────
    VOCAB_PATH   = PATHS["vocab"]
    NUM_ARTICLES = TOKENIZER["vocabArticles"]
    MIN_FREQ     = TOKENIZER["minFrequency"]
    SEQ_LENGTH   = TOKENIZER["seqLengthForTest"]

    # ── Step 1: Load data ─────────────────────────────────────────────────────
    print("=" * 60)
    print(f"  STEP 1: Loading Data ({DATA['source']})")
    print("=" * 60)

    if os.path.exists(VOCAB_PATH):
        print(f"  Found existing vocab at '{VOCAB_PATH}', loading it instead.\n")
        tokenizer = WordTokenizer.load(VOCAB_PATH)
    else:
        if DATA["source"] == "claude-opus":
            texts = loadClaudeOpusDataset(
                datasetName=DATA["claudeDataset"],
                numRows=NUM_ARTICLES,
            )
        else:
            texts = loadWikipediaDataset(numArticles=NUM_ARTICLES)

        # ── Step 2: Build vocabulary ──────────────────────────────────────────
        print("=" * 60)
        print("  STEP 2: Building Vocabulary")
        print("=" * 60)

        tokenizer = WordTokenizer(minFrequency=MIN_FREQ)
        tokenizer.buildVocab(extractTextsForVocab(texts))

        # ── Step 3: Save vocabulary ───────────────────────────────────────────
        print("=" * 60)
        print("  STEP 3: Saving Vocabulary")
        print("=" * 60)

        tokenizer.save(VOCAB_PATH)

    # ── Step 4: Test encode / decode ──────────────────────────────────────────
    print("=" * 60)
    print("  STEP 4: Testing Encode / Decode")
    print("=" * 60)

    testSentence = "The Transformer model was introduced in the paper Attention Is All You Need"
    print(f"\n  Input:    \"{testSentence}\"")

    encoded = tokenizer.encode(testSentence, maxLength=SEQ_LENGTH)
    print(f"  Encoded:  {encoded[:20]}...  (showing first 20 of {len(encoded)} IDs)")

    decoded = tokenizer.decode(encoded)
    print(f"  Decoded:  \"{decoded}\"")

    # Show vocab size — this is what you pass to buildTransformer()
    print(f"\n  ┌─────────────────────────────────────────────┐")
    print(f"  │  vocabSize = {tokenizer.vocabSize:<10,}                    │")
    print(f"  │  Use this value in buildTransformer():       │")
    print(f"  │    source_vocabSize = {tokenizer.vocabSize:<10,}           │")
    print(f"  │    target_vocabSize = {tokenizer.vocabSize:<10,}           │")
    print(f"  └─────────────────────────────────────────────┘")
    print()
