# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  generate.py  –  Use your trained Transformer to generate text             ║
# ║                                                                            ║
# ║  USAGE:                                                                    ║
# ║    python3 generate.py                              (interactive mode)     ║
# ║    python3 generate.py --prompt "The history of"    (single prompt)        ║
# ║    python3 generate.py --resume                     (continue training)    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import torch
import argparse
import os
import sys

from model import buildTransformer
from tokenizer import WordTokenizer, tokenize
from config import MODEL, PATHS

# ── Pull settings from config.py ─────────────────────────────────────────────
MODEL_CONFIG = MODEL
VOCAB_PATH = PATHS["vocab"]
CHECKPOINT_PATH = PATHS["checkpoint"]

# Special token IDs
PAD_ID = 0
SOS_ID = 2
EOS_ID = 3


def loadTrainedModel(vocabPath, checkpointPath, device):
    """
    Loads the tokenizer and the trained model weights from disk.

    What's in the checkpoint file (checkpoint.pt)?
      - model_state_dict:      All the learned weights (embeddings, attention
                                matrices, feed-forward layers, etc.)
      - optimizer_state_dict:  Optimizer state (needed only for resuming training)
      - epoch:                 Which epoch the checkpoint was saved at
      - val_loss:              The validation loss at that point

    The model architecture must match EXACTLY what was used during training —
    same dModel, dFF, h, N, vocabSize — otherwise the weights won't fit.
    """
    # Load tokenizer
    tokenizer = WordTokenizer.load(vocabPath)

    # Rebuild the same model architecture (empty weights)
    model = buildTransformer(
        source_vocabSize=tokenizer.vocabSize,
        target_vocabSize=tokenizer.vocabSize,
        source_sequenceLength=MODEL_CONFIG["seqLength"],
        target_sequenceLength=MODEL_CONFIG["seqLength"],
        N=MODEL_CONFIG["N"],
        dModel=MODEL_CONFIG["dModel"],
        dFF=MODEL_CONFIG["dFF"],
        h=MODEL_CONFIG["h"],
        dropout=MODEL_CONFIG["dropout"],
    )

    # Load trained weights into the model
    checkpoint = torch.load(checkpointPath, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()  # ... disable dropout for inference

    epoch = checkpoint.get("epoch", "?")
    valLoss = checkpoint.get("val_loss", "?")
    print(f"  Loaded checkpoint from epoch {epoch} (val_loss={valLoss})")

    return model, tokenizer


def generate(model, tokenizer, prompt, maxTokens=256, device="cpu"):
    """
    Generates text by feeding a prompt through the encoder, then having
    the decoder predict one token at a time (autoregressive generation).

    How autoregressive generation works:
      1. Encode the prompt → encoder output (context)
      2. Start decoder with just [SOS]
      3. Predict the next token
      4. Append that token to the decoder input
      5. Repeat steps 3-4 until [EOS] or maxTokens reached

    This is how ChatGPT, etc. work — one token at a time.
    """
    sl = MODEL_CONFIG["seqLength"]

    # ── Tokenize the prompt for the encoder ───────────────────────────────
    words = tokenize(prompt)
    unkId = tokenizer.word2id.get("<UNK>", 1)
    srcIds = [tokenizer.word2id.get(w, unkId) for w in words]

    # Truncate or pad to seqLength
    srcIds = srcIds[:sl]
    srcIds = srcIds + [PAD_ID] * (sl - len(srcIds))

    # Convert to tensor
    encoderInput = torch.tensor([srcIds], dtype=torch.long, device=device)

    # Encoder mask (1 where not padding)
    encoderMask = (encoderInput != PAD_ID).unsqueeze(1).unsqueeze(1).int()

    # ── Encode the prompt ─────────────────────────────────────────────────
    with torch.no_grad():
        encoderOut = model.encode(encoderInput, encoderMask)

    # ── Decode token by token ─────────────────────────────────────────────
    decoderIds = [SOS_ID]  # ... start with <SOS>

    for _ in range(maxTokens):
        # Pad decoder input to seqLength
        decInput = decoderIds + [PAD_ID] * (sl - len(decoderIds))
        decInput = decInput[:sl]
        decoderInput = torch.tensor([decInput], dtype=torch.long, device=device)

        # Causal mask: decoder can only see past tokens
        decLen = len(decoderIds)
        causalMask = torch.tril(torch.ones(sl, sl, dtype=torch.int, device=device))
        decoderMask = causalMask.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            decoderOut = model.decode(encoderOut, encoderMask, decoderInput, decoderMask)
            logits = model.projection(decoderOut)

        # Get the predicted token at the last position
        nextTokenLogits = logits[0, decLen - 1, :]  # ... logits for next position

        # Greedy decoding: pick the highest probability token
        nextToken = nextTokenLogits.argmax().item()

        # Stop if model predicts <EOS>
        if nextToken == EOS_ID:
            break

        decoderIds.append(nextToken)

        # Stop if we've filled the sequence
        if len(decoderIds) >= sl:
            break

    # ── Decode token IDs back to words ────────────────────────────────────
    outputWords = []
    for tid in decoderIds[1:]:  # ... skip <SOS>
        if tid in (PAD_ID, EOS_ID, SOS_ID):
            continue
        word = tokenizer.id2word.get(tid, "<UNK>")
        outputWords.append(word)

    return " ".join(outputWords)


def interactive(model, tokenizer, device):
    """Interactive prompt loop — type a prompt, see what the model generates."""
    print("\n" + "=" * 60)
    print("  🧠 Transformer Text Generation")
    print("  Type a prompt and press Enter. Type 'quit' to exit.")
    print("=" * 60)

    while True:
        try:
            prompt = input("\n  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break

        if not prompt or prompt.lower() in ("quit", "exit", "q"):
            print("  Goodbye!")
            break

        output = generate(model, tokenizer, prompt, maxTokens=256, device=device)
        print(f"  Model: {output}")


def main():
    parser = argparse.ArgumentParser(description="Generate text with trained Transformer")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt to generate from")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH)
    parser.add_argument("--vocab", type=str, default=VOCAB_PATH)
    args = parser.parse_args()

    # Check files exist
    if not os.path.exists(args.vocab):
        print(f"  ❌ Vocab not found: {args.vocab}")
        print(f"     Run 'python3 tokenizer.py' first.")
        sys.exit(1)
    if not os.path.exists(args.checkpoint):
        print(f"  ❌ Checkpoint not found: {args.checkpoint}")
        print(f"     Run 'python3 train.py' first to train the model.")
        sys.exit(1)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"  Device: {device}")

    # Load model
    model, tokenizer = loadTrainedModel(args.vocab, args.checkpoint, device)

    if args.prompt:
        output = generate(model, tokenizer, args.prompt, args.max_tokens, device=device)
        print(f"\n  Prompt:    {args.prompt}")
        print(f"  Generated: {output}")
    else:
        interactive(model, tokenizer, device)


if __name__ == "__main__":
    main()
