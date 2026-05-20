# train.py — Training & Validation Loop
#
# Run training:    python3 train.py
# Run dashboard:   python3 dashboard.py   (separate process)

import torch
import torch.nn as nn
import math
import os
import json
import time
try:
    import psutil
except ImportError:
    psutil = None
import resource
from torch.utils.data import Dataset, DataLoader

from model import buildTransformer
from tokenizer import WordTokenizer, tokenize, loadWikipediaDataset, loadClaudeOpusDataset, extractTextsForVocab
from config import MODEL, TRAINING, DATA, PATHS, RESUME, DASHBOARD
from autotuner import ActiveConfigMatcher

# Flatten config for easy access
CONFIG = {
    **MODEL,
    "epochs": TRAINING["epochs"],
    "batchSize": TRAINING["batchSize"],
    "maxLR": TRAINING["maxLR"],
    "minLR": TRAINING["minLR"],
    "warmupSteps": TRAINING["warmupSteps"],
    "labelSmoothing": TRAINING["labelSmoothing"],
    "gradClipNorm": TRAINING["gradClipNorm"],
    "logInterval": TRAINING.get("logInterval", 25),
    "accumSteps": TRAINING.get("accumSteps", 1),
    "numArticles": DATA["numArticles"],
    "valSplit": DATA["valSplit"],
    "vocabPath": PATHS["vocab"],
    "metricsPath": PATHS["metrics"],
    "checkpointPath": PATHS["checkpoint"],
    "dashboardPort": DASHBOARD["port"],
    "totalSteps": 0,
}

PAD_ID = 0
UNK_ID = 1
SOS_ID = 2
EOS_ID = 3


# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET — Converts text into (encoder, decoder) training pairs
#
#  Supports two modes:
#    1. DPO pairs:     list of (prompt, response) tuples  → encoder=prompt, decoder=response
#    2. Plain text:    list of strings  → sliding-window split into src/tgt halves
# ═══════════════════════════════════════════════════════════════════════════════

class TransformerDataset(Dataset):
    """
    Builds encoder/decoder training pairs.

    If `texts` contains (prompt, response) tuples:
      encoder_input = tokenized prompt (truncated/padded to seqLength)
      decoder_input = [SOS] + tokenized response[:-1]
      label         = tokenized response[:-1] + [EOS]

    If `texts` contains plain strings:
      Uses a sliding window split in half (legacy mode for Wikipedia, etc.).
    """

    def __init__(self, texts, tokenizer, seqLength):
        self.tokenizer = tokenizer
        self.seqLength = seqLength
        self.samples = []

        unkId = tokenizer.word2id.get("<UNK>", UNK_ID)

        # Detect mode: list of tuples (prompt, response) vs list of strings
        if texts and isinstance(texts[0], (list, tuple)) and len(texts[0]) == 2:
            # ── DPO pair mode ─────────────────────────────────────────────
            for prompt, response in texts:
                srcWords = tokenize(prompt)
                tgtWords = tokenize(response)
                if not srcWords or not tgtWords:
                    continue
                src = [tokenizer.word2id.get(w, unkId) for w in srcWords]
                tgt = [tokenizer.word2id.get(w, unkId) for w in tgtWords]
                # Truncate to seqLength
                src = src[:seqLength]
                tgt = tgt[:seqLength - 1]  # leave room for SOS/EOS
                # Pad source to seqLength
                src = src + [PAD_ID] * (seqLength - len(src))
                # Pad target to seqLength - 1
                tgt = tgt + [PAD_ID] * (seqLength - 1 - len(tgt))
                self.samples.append((src, tgt))
        else:
            # ── Sliding window mode (plain text) ──────────────────────────
            for text in texts:
                words = tokenize(text)
                ids = [tokenizer.word2id.get(w, unkId) for w in words]
                windowSize = seqLength * 2
                for i in range(0, len(ids) - windowSize, seqLength):
                    src = ids[i : i + seqLength]
                    tgt = ids[i + seqLength : i + windowSize]
                    self.samples.append((src, tgt))

        print(f"  Created {len(self.samples):,} training samples "
              f"from {len(texts):,} articles")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src_ids, tgt_ids = self.samples[idx]
        sl = self.seqLength

        encoder_input = torch.tensor(src_ids[:sl], dtype=torch.long)
        decoder_input = torch.tensor([SOS_ID] + tgt_ids[:sl - 1], dtype=torch.long)
        label = torch.tensor(tgt_ids[:sl - 1] + [EOS_ID], dtype=torch.long)

        # Masks: encoder pads + decoder causal mask
        encoder_mask = (encoder_input != PAD_ID).unsqueeze(0).unsqueeze(0).int()
        decoder_padding = (decoder_input != PAD_ID).unsqueeze(0).int()
        causal_mask = torch.tril(torch.ones(sl, sl, dtype=torch.int))
        decoder_mask = decoder_padding & causal_mask

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": encoder_mask,
            "decoder_mask": decoder_mask.unsqueeze(0),
            "label": label,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  LEARNING RATE — Cosine annealing with linear warmup
# ═══════════════════════════════════════════════════════════════════════════════

def getLearningRate(step, maxLR, minLR, warmupSteps, totalSteps):
    """
    Phase 1 (warmup):  LR climbs linearly from 0 → maxLR
    Phase 2 (decay):   LR follows a cosine curve from maxLR → minLR
    """
    if step < warmupSteps:
        return maxLR * (step / max(warmupSteps, 1))
    progress = (step - warmupSteps) / max(totalSteps - warmupSteps, 1)
    progress = min(progress, 1.0)
    return minLR + (maxLR - minLR) * 0.5 * (1.0 + math.cos(math.pi * progress))


# ═══════════════════════════════════════════════════════════════════════════════
#  METRICS TRACKER — Feeds data to the dashboard
# ═══════════════════════════════════════════════════════════════════════════════

class MetricsTracker:
    def __init__(self, config, metricsPath):
        self.path = metricsPath
        self.startTime = time.time()
        self.data = {
            "config": config,
            "status": "initializing",
            "currentEpoch": 0,
            "totalEpochs": config["epochs"],
            "currentBatch": 0,
            "totalBatches": 0,
            "globalStep": 0,
            "trainLosses": [],
            "smoothedLoss": [],
            "valLosses": [],
            "learningRates": [],
            "gradNorms": [],
            "trainAccuracies": [],
            "epochTrainLosses": [],
            "epochValAccuracies": [],
            "perplexities": [],
            "attentionWeights": [],
            "attentionSrcTokens": [],
            "attentionTgtTokens": [],
            "predictions": [],
            "embeddingNodes": [],
            "embeddingEdges": [],
            "tokensPerSecond": 0,
            "elapsedSeconds": 0,
            "modelParams": 0,
            "batchTime": 0,
            "memoryUsage": 0,
            "attnEntropy": 0,
        }
        self.save()

    def update(self, **kwargs):
        self.data.update(kwargs)

    def save(self):
        try:
            with open(self.path, "w") as f:
                json.dump(self.data, f)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINING LOOP — One epoch
# ═══════════════════════════════════════════════════════════════════════════════

def trainOneEpoch(model, dataloader, optimizer, criterion, device, tokenizer, metrics, epoch, tuner=None):
    model.train()
    totalLoss = 0
    batchCount = len(dataloader)
    startTime = time.time()
    tokenCount = 0
    accumSteps = CONFIG.get("accumSteps", 1)

    # Determine autocast dtype for MPS / CUDA / CPU
    useMPS = (device.type == "mps")
    useCUDA = (device.type == "cuda")
    castDevice = "mps" if useMPS else ("cuda" if useCUDA else "cpu")
    castDtype = torch.float16 if (useMPS or useCUDA) else torch.bfloat16

    optimizer.zero_grad()  # zero once; accumulate across micro-batches

    for batchIdx, batch in enumerate(dataloader):
        enc_input = batch["encoder_input"].to(device)
        dec_input = batch["decoder_input"].to(device)
        enc_mask = batch["encoder_mask"].to(device)
        dec_mask = batch["decoder_mask"].to(device)
        label = batch["label"].to(device)

        # Forward with mixed-precision autocast
        with torch.autocast(device_type=castDevice, dtype=castDtype):
            encoderOut = model.encode(enc_input, enc_mask)
            decoderOut = model.decode(encoderOut, enc_mask, dec_input, dec_mask)
            projOut = model.projection(decoderOut)
            loss = criterion(projOut.view(-1, projOut.size(-1)), label.view(-1))
            loss = loss / accumSteps  # scale loss for gradient accumulation

        loss.backward()

        # Step optimizer every accumSteps micro-batches
        # Use tuner's effective grad clip if active, otherwise use config default
        activeGradClip = tuner.effectiveGradClip if tuner else CONFIG["gradClipNorm"]
        if (batchIdx + 1) % accumSteps == 0 or (batchIdx + 1) == batchCount:
            gradNorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=activeGradClip)
            optimizer.step()
            optimizer.zero_grad()
        else:
            gradNorm = torch.tensor(0.0)  # placeholder for non-step batches

        # Update learning rate (cosine schedule + tuner scaling)
        globalStep = metrics.data["globalStep"] + 1
        lr = getLearningRate(globalStep, CONFIG["maxLR"], CONFIG["minLR"],
                            CONFIG["warmupSteps"], CONFIG["totalSteps"])
        if tuner:
            lr = tuner.scaleLR(lr)
        for paramGroup in optimizer.param_groups:
            paramGroup["lr"] = lr

        # Token accuracy (ignoring PAD positions)
        with torch.no_grad():
            predicted = projOut.argmax(dim=-1)
            nonPadMask = (label != PAD_ID)
            correct = (predicted == label) & nonPadMask
            accuracy = correct.sum().item() / max(nonPadMask.sum().item(), 1)

        # Track metrics (unscale loss for logging)
        batchLoss = loss.item() * accumSteps
        totalLoss += batchLoss
        tokenCount += enc_input.numel()
        elapsedTotal = time.time() - startTime
        tokPerSec = tokenCount / max(elapsedTotal, 1)

        # Exponential moving average of loss
        prevSmoothed = metrics.data["smoothedLoss"]
        ema = 0.95 * prevSmoothed[-1] + 0.05 * batchLoss if prevSmoothed else batchLoss

        metrics.data["trainLosses"].append(round(batchLoss, 4))
        metrics.data["smoothedLoss"].append(round(ema, 4))
        metrics.data["learningRates"].append(round(lr, 8))
        metrics.data["gradNorms"].append(round(gradNorm.item(), 4))
        metrics.data["trainAccuracies"].append(round(accuracy, 4))

        # ── Auto-tuner: analyze and adjust ────────────────────────────────
        if tuner:
            tunerAdj = tuner.step(batchIdx, batchLoss, gradNorm.item(), accuracy, lr, metrics.data)
            # If tuner adjusted label smoothing, rebuild criterion
            if "labelSmoothing" in tunerAdj:
                criterion.label_smoothing = tuner.effectiveLabelSmoothing
            # If tuner adjusted dropout, apply it live to the model
            if "dropout" in tunerAdj:
                import torch.nn as nn
                for m in model.modules():
                    if isinstance(m, nn.Dropout):
                        m.p = tuner.effectiveDropout

        # Periodic dashboard + console update
        if (batchIdx + 1) % CONFIG["logInterval"] == 0 or batchIdx == batchCount - 1:
            if psutil:
                mem = psutil.Process().memory_info().rss / (1024**3)
            else:
                mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)  # bytes → GB on macOS

            captureAttention(model, enc_input, dec_input, tokenizer, metrics)
            capturePredictions(projOut, enc_input, label, tokenizer, metrics)

            # Capture embeddings once per epoch (first log interval of each epoch)
            if batchIdx + 1 == CONFIG["logInterval"]:
                captureEmbeddings(model, tokenizer, metrics)

            metrics.update(
                currentBatch=batchIdx + 1,
                totalBatches=batchCount,
                globalStep=globalStep,
                tokensPerSecond=round(tokPerSec),
                status="training",
                memoryUsage=mem,
                elapsedSeconds=int(time.time() - metrics.startTime),
                batchTime=elapsedTotal / (batchIdx + 1)
            )
            metrics.save()
            print(f"  Epoch {epoch+1} | Batch {batchIdx+1}/{batchCount} "
                  f"| Loss: {batchLoss:.4f} | LR: {lr:.6f} "
                  f"| {tokPerSec:,.0f} tok/s")
        else:
            metrics.update(currentBatch=batchIdx + 1, totalBatches=batchCount, globalStep=globalStep)

    return totalLoss / max(batchCount, 1)


# ═══════════════════════════════════════════════════════════════════════════════
#  VALIDATION LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def validate(model, dataloader, criterion, device, tokenizer, metrics):
    model.eval()
    totalLoss = 0
    totalCorrect = 0
    totalTokens = 0
    batchCount = len(dataloader)

    with torch.no_grad():
        for batch in dataloader:
            enc_input = batch["encoder_input"].to(device)
            dec_input = batch["decoder_input"].to(device)
            enc_mask = batch["encoder_mask"].to(device)
            dec_mask = batch["decoder_mask"].to(device)
            label = batch["label"].to(device)

            encoderOut = model.encode(enc_input, enc_mask)
            decoderOut = model.decode(encoderOut, enc_mask, dec_input, dec_mask)
            projOut = model.projection(decoderOut)

            loss = criterion(projOut.view(-1, projOut.size(-1)), label.view(-1))
            totalLoss += loss.item()

            predicted = projOut.argmax(dim=-1)
            nonPadMask = (label != PAD_ID)
            totalCorrect += ((predicted == label) & nonPadMask).sum().item()
            totalTokens += nonPadMask.sum().item()

    avgLoss = totalLoss / max(batchCount, 1)
    valAccuracy = totalCorrect / max(totalTokens, 1)
    metrics.data["epochValAccuracies"].append(round(valAccuracy, 4))
    return avgLoss


# ═══════════════════════════════════════════════════════════════════════════════
#  TELEMETRY CAPTURES — attention, predictions, embeddings for dashboard
# ═══════════════════════════════════════════════════════════════════════════════

def captureAttention(model, enc_input, dec_input, tokenizer, metrics):
    """Extracts per-layer, per-head attention weights for the dashboard explorer."""
    try:
        # ── Per-layer per-head attention for the Attention Explorer tab ──
        allLayerAttentions = []
        perHeadEntropies = []
        for layer in model.encoder.layers:
            layerAttn = layer.selfAttentionBlock.attention_scores  # (B, h, S, S)
            headAttn = layerAttn[0]  # first batch item: (h, S, S)
            size = min(32, headAttn.size(-1))
            headsList = []
            for headIdx in range(headAttn.size(0)):
                headSlice = headAttn[headIdx, :size, :size]
                headsList.append([[round(v, 4) for v in row] for row in headSlice.cpu().tolist()])
                # Per-head entropy
                hEnt = -(headSlice * torch.log(headSlice + 1e-9)).sum(dim=-1).mean().item()
                perHeadEntropies.append(round(hEnt, 4))
            allLayerAttentions.append(headsList)

        # ── Legacy: averaged heatmap from layer 0 (backward compat) ─────
        attn = model.encoder.layers[0].selfAttentionBlock.attention_scores
        attn = attn[0].mean(dim=0)  # average across heads
        size = min(32, attn.size(0))
        attnSlice = attn[:size, :size].cpu().tolist()

        srcIds = enc_input[0][:size].cpu().tolist()
        srcTokens = [tokenizer.id2word.get(i, "?") for i in srcIds]

        # Global attention entropy
        entropy = -(attn * torch.log(attn + 1e-9)).sum(dim=-1).mean().item()

        metrics.update(
            attentionWeights=[[round(v, 4) for v in row] for row in attnSlice],
            attentionSrcTokens=srcTokens,
            attentionTgtTokens=srcTokens,
            attnEntropy=entropy,
            allLayerAttentions=allLayerAttentions,
            perHeadEntropies=perHeadEntropies,
        )
    except Exception:
        pass


def capturePredictions(projOut, enc_input, label, tokenizer, metrics):
    """Grabs sample predictions for the dashboard."""
    predicted = projOut.argmax(dim=-1)
    samples = []
    for i in range(min(10, predicted.size(0))):
        samples.append({
            "source": tokenizer.decode(enc_input[i].cpu().tolist()),
            "target": tokenizer.decode(label[i].cpu().tolist()),
            "predicted": tokenizer.decode(predicted[i].cpu().tolist()),
        })
    metrics.update(predictions=samples)


def captureEmbeddings(model, tokenizer, metrics, numWords=80, kNeighbors=3):
    """Projects learned embeddings into 3D via PCA for the word-map visualization."""
    try:
        embMatrix = model.sourceEmbed.embedding.weight.detach().cpu()

        maxId = min(numWords + 4, embMatrix.size(0))
        ids = list(range(4, maxId))
        subset = embMatrix[ids]
        words = [tokenizer.id2word.get(i, "?") for i in ids]

        # PCA via SVD → 3D
        centered = subset - subset.mean(dim=0)
        U, S, V = torch.svd(centered)
        coords = (centered @ V[:, :3]).tolist()

        # Normalize to [-1, 1]
        xs, ys, zs = [c[0] for c in coords], [c[1] for c in coords], [c[2] for c in coords]
        xMin, xMax = min(xs), max(xs)
        yMin, yMax = min(ys), max(ys)
        zMin, zMax = min(zs), max(zs)
        xR = max(xMax - xMin, 1e-6)
        yR = max(yMax - yMin, 1e-6)
        zR = max(zMax - zMin, 1e-6)

        nodes = []
        for i, word in enumerate(words):
            nx = (coords[i][0] - xMin) / xR * 2 - 1
            ny = (coords[i][1] - yMin) / yR * 2 - 1
            nz = (coords[i][2] - zMin) / zR * 2 - 1
            nodes.append({"word": word, "x": round(nx, 4), "y": round(ny, 4), "z": round(nz, 4)})

        # k nearest neighbors via cosine similarity
        norms = subset.norm(dim=1, keepdim=True).clamp(min=1e-8)
        similarity = (subset / norms) @ (subset / norms).T

        edges = []
        for i in range(len(ids)):
            sim_row = similarity[i].clone()
            sim_row[i] = -1
            _, topk = sim_row.topk(kNeighbors)
            for j in topk.tolist():
                if [j, i] not in edges:
                    edges.append([i, j])

        metrics.update(embeddingNodes=nodes, embeddingEdges=edges)
        print(f"  📐 Captured {len(nodes)} embedding nodes for visualization")
    except Exception as e:
        print(f"  ⚠️  captureEmbeddings failed: {e}")





# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  🧠 TRANSFORMER TRAINING")
    print("=" * 60)

    # Device selection: MPS (Apple Silicon) > CUDA (NVIDIA) > CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("  Device: Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("  Device: NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("  Device: CPU (training will be slow)")

    # Load tokenizer
    print(f"\n  Loading tokenizer from {CONFIG['vocabPath']}...")
    tokenizer = WordTokenizer.load(CONFIG["vocabPath"])

    # Run tokenizer self-check before training starts
    tokenizer.selfCheck()

    # Load data
    print(f"\n  Data source: {DATA['source']}")
    if DATA["source"] == "claude-opus":
        print(f"  Loading {CONFIG['numArticles']:,} rows from DPO dataset...")
        texts = loadClaudeOpusDataset(datasetName=DATA["claudeDataset"], numRows=CONFIG["numArticles"])
    else:
        print(f"  Loading {CONFIG['numArticles']:,} Wikipedia articles...")
        texts = loadWikipediaDataset(numArticles=CONFIG["numArticles"])

    # Train/val split
    splitIdx = int(len(texts) * (1 - CONFIG["valSplit"]))
    trainTexts, valTexts = texts[:splitIdx], texts[splitIdx:]
    print(f"  Train samples: {len(trainTexts):,}")
    print(f"  Val samples:   {len(valTexts):,}")

    # Build datasets and dataloaders
    print(f"\n  Building training dataset...")
    trainDataset = TransformerDataset(trainTexts, tokenizer, CONFIG["seqLength"])
    valDataset = TransformerDataset(valTexts, tokenizer, CONFIG["seqLength"])

    trainLoader = DataLoader(trainDataset, batch_size=CONFIG["batchSize"], shuffle=True, drop_last=True)
    valLoader = DataLoader(valDataset, batch_size=CONFIG["batchSize"], shuffle=False, drop_last=True)

    # Build model
    print(f"\n  Building Transformer model...")
    model = buildTransformer(
        source_vocabSize=tokenizer.vocabSize,
        target_vocabSize=tokenizer.vocabSize,
        source_sequenceLength=CONFIG["seqLength"],
        target_sequenceLength=CONFIG["seqLength"],
        N=CONFIG["N"],
        dModel=CONFIG["dModel"],
        dFF=CONFIG["dFF"],
        h=CONFIG["h"],
        dropout=CONFIG["dropout"],
    ).to(device)

    paramCount = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {paramCount:,}")

    CONFIG["totalSteps"] = len(trainLoader) * CONFIG["epochs"]
    print(f"  Total training steps: {CONFIG['totalSteps']:,}")

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["maxLR"], betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=CONFIG["labelSmoothing"])

    # Resume from checkpoint if available
    startEpoch = 0
    bestValLoss = float("inf")
    if RESUME["enabled"] and os.path.exists(CONFIG["checkpointPath"]):
        print(f"\n  📂 Found checkpoint: {CONFIG['checkpointPath']}")
        checkpoint = torch.load(CONFIG["checkpointPath"], map_location=device, weights_only=True)
        try:
            model.load_state_dict(checkpoint["model_state_dict"])
            if RESUME["loadOptimizer"]:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                print("  Loaded optimizer state (full resume)")
            else:
                print("  Fresh optimizer (fine-tune mode)")
            startEpoch = checkpoint.get("epoch", 0) + 1
            bestValLoss = checkpoint.get("val_loss", float("inf"))
            print(f"  Resuming from epoch {startEpoch} (best val_loss={bestValLoss:.4f})")
        except RuntimeError as e:
            print(f"  ⚠️  Checkpoint architecture mismatch. Ignoring checkpoint and starting fresh.")
            print(f"  🆕 Training from scratch.")
    else:
        print(f"\n  🆕 Training from scratch.")

    # Metrics (dashboard reads metrics.json from a separate process)
    metrics = MetricsTracker(CONFIG, CONFIG["metricsPath"])
    metrics.update(modelParams=paramCount)
    metrics.save()
    print(f"\n  📊 Run 'python3 dashboard.py' in another terminal for live stats.")

    # ── Active Configuration Matcher (auto-tuner) ─────────────────────────
    tuner = ActiveConfigMatcher(CONFIG, metrics, enabled=True)
    print(f"  🔧 Auto-tuner enabled (monitoring gradients, loss, accuracy)")

    # ── Training loop ─────────────────────────────────────────────────────
    startTime = time.time()

    for epoch in range(startEpoch, CONFIG["epochs"]):
        print(f"\n{'─' * 60}")
        print(f"  EPOCH {epoch + 1} / {CONFIG['epochs']}")
        print(f"{'─' * 60}")

        metrics.update(currentEpoch=epoch + 1, status="training")
        metrics.save()

        avgTrainLoss = trainOneEpoch(model, trainLoader, optimizer, criterion, device, tokenizer, metrics, epoch, tuner=tuner)

        metrics.update(status="validating")
        metrics.save()
        avgValLoss = validate(model, valLoader, criterion, device, tokenizer, metrics)

        perplexity = math.exp(min(avgValLoss, 20))
        elapsed = time.time() - startTime

        metrics.data["epochTrainLosses"].append(round(avgTrainLoss, 4))
        metrics.data["valLosses"].append(round(avgValLoss, 4))
        metrics.data["perplexities"].append(round(perplexity, 2))
        metrics.update(elapsedSeconds=round(elapsed))
        metrics.save()

        print(f"\n  ── Epoch {epoch+1} Summary ──")
        print(f"     Train Loss:  {avgTrainLoss:.4f}")
        print(f"     Val Loss:    {avgValLoss:.4f}")
        print(f"     Perplexity:  {perplexity:.2f}")
        print(f"     Elapsed:     {elapsed/60:.1f} min")

        if avgValLoss < bestValLoss:
            bestValLoss = avgValLoss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avgValLoss,
            }, CONFIG["checkpointPath"])
            print(f"     ✅ Best model saved! (val_loss={avgValLoss:.4f})")

    # Post-training tokenizer diagnostics
    tokenizer.diagnose()

    # Auto-tuner summary
    tuner.summary()

    metrics.update(status="complete")
    metrics.save()
    print(f"\n{'=' * 60}")
    print(f"  ✅ Training complete!")
    print(f"  Final metrics saved to {CONFIG['metricsPath']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
