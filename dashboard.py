# dashboard.py — Standalone dashboard server
#
# Run separately from training:
#   python3 dashboard.py
#
# This decouples the HTTP server from the GPU training loop,
# so serving dashboard requests never steals compute from training.

import json
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from config import PATHS, DASHBOARD

METRICS_PATH = PATHS["metrics"]
VOCAB_PATH = PATHS["vocab"]
PORT = DASHBOARD["port"]

# Load vocabulary once at startup for the token visualizer API
_vocab = None
_vocab_config = None

def getVocab():
    global _vocab, _vocab_config
    if _vocab is None and os.path.exists(VOCAB_PATH):
        with open(VOCAB_PATH, "r") as f:
            data = json.load(f)
        _vocab = {
            "word2id": data["word2id"],
            "id2word": {int(k): v for k, v in data["id2word"].items()},
            "vocabSize": data["vocabSize"],
        }
        _vocab_config = data.get("config", None)
    return _vocab


def getVocabConfig():
    """Returns the TokenizerConfig dict stored inside vocab.json."""
    getVocab()  # ensure loaded
    return _vocab_config


def reloadVocab():
    """Force-reload vocab from disk (after a config change or rebuild)."""
    global _vocab, _vocab_config
    _vocab = None
    _vocab_config = None
    return getVocab()


def tokenizeForViz(text, vocab):
    """Tokenize text and return per-token info for the visualizer.
    Uses the same tokenize() function as the training pipeline."""
    from tokenizer import tokenize as realTokenize

    words = realTokenize(text)

    unkId = vocab["word2id"].get("<UNK>", 1)
    sosId = vocab["word2id"].get("<SOS>", 2)
    eosId = vocab["word2id"].get("<EOS>", 3)

    tokens = [{"token": "<SOS>", "id": sosId, "type": "special"}]
    for word in words:
        wid = vocab["word2id"].get(word, unkId)
        ttype = "unknown" if wid == unkId else "known"
        tokens.append({"token": word, "id": wid, "type": ttype})
    tokens.append({"token": "<EOS>", "id": eosId, "type": "special"})

    known = sum(1 for t in tokens if t["type"] == "known")
    unknown = sum(1 for t in tokens if t["type"] == "unknown")
    total = known + unknown
    unkRate = unknown / max(total, 1)

    return {
        "tokens": tokens,
        "stats": {
            "totalTokens": len(tokens),
            "knownWords": known,
            "unknownWords": unknown,
            "unkRate": round(unkRate, 4),
            "vocabSize": vocab["vocabSize"],
        }
    }


class DashboardHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.path = "/dashboard.html"
        elif self.path == "/api/metrics":
            self._serveJSON(METRICS_PATH)
            return
        elif self.path == "/api/autotuner":
            self._handleAutotuner()
            return
        elif self.path.startswith("/api/tokenize?"):
            self._handleTokenize()
            return
        elif self.path == "/api/tokenizer/config":
            self._handleGetTokenizerConfig()
            return
        elif self.path == "/api/tokenizer/encodings":
            self._handleGetEncodings()
            return
        return super().do_GET()

    def do_POST(self):
        if self.path == "/api/tokenizer/config":
            self._handlePostTokenizerConfig()
            return
        elif self.path == "/api/tokenizer/rebuild":
            self._handleRebuildVocab()
            return
        self.send_error(404)

    def _serveJSON(self, filepath):
        try:
            with open(filepath, "r") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data.encode())
        except FileNotFoundError:
            self._sendJSON(404, {"error": "not found"})
        except Exception:
            self._sendJSON(500, {"error": "server error"})

    def _handleTokenize(self):
        query = parse_qs(urlparse(self.path).query)
        text = query.get("text", [""])[0]
        vocab = getVocab()
        if not vocab:
            self._sendJSON(503, {"error": "vocab not loaded"})
            return
        result = tokenizeForViz(text, vocab)
        self._sendJSON(200, result)

    def _handleAutotuner(self):
        """Serve the auto-tuner state from metrics.json."""
        try:
            with open(METRICS_PATH, "r") as f:
                data = json.load(f)
            tunerState = data.get("autotuner", {
                "enabled": False,
                "healthStatus": "unknown",
                "healthReasons": [],
                "totalAdjustments": 0,
                "recentAdjustments": [],
                "diagnostics": {},
            })
            self._sendJSON(200, tunerState)
        except FileNotFoundError:
            self._sendJSON(503, {"error": "metrics not available"})
        except Exception:
            self._sendJSON(500, {"error": "server error"})

    # ── Tokenizer Config Endpoints ────────────────────────────────────────

    def _handleGetTokenizerConfig(self):
        """GET /api/tokenizer/config — return current tokenizer settings."""
        config = getVocabConfig()
        if config is None:
            # Fall back to defaults if no config stored in vocab.json
            from tokenizer import TokenizerConfig
            config = TokenizerConfig().to_dict()
        self._sendJSON(200, config)

    def _handleGetEncodings(self):
        """GET /api/tokenizer/encodings — list available encoding presets."""
        from tokenizer import list_encodings
        self._sendJSON(200, list_encodings())

    def _handlePostTokenizerConfig(self):
        """POST /api/tokenizer/config — update tokenizer config in vocab.json."""
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}

            if not os.path.exists(VOCAB_PATH):
                self._sendJSON(404, {"error": "vocab.json not found — build vocab first"})
                return

            # Read existing vocab.json
            with open(VOCAB_PATH, "r") as f:
                vocabData = json.load(f)

            # Merge new config fields into existing config
            existingConfig = vocabData.get("config", {})
            existingConfig.update(body)

            # Validate using TokenizerConfig
            from tokenizer import TokenizerConfig
            newConfig = TokenizerConfig.from_dict(existingConfig)
            warnings = newConfig.validate()

            # Persist back
            vocabData["config"] = newConfig.to_dict()
            vocabData["minFrequency"] = newConfig.min_frequency  # legacy compat

            with open(VOCAB_PATH, "w") as f:
                json.dump(vocabData, f)

            # Reload in-memory state
            reloadVocab()

            self._sendJSON(200, {
                "status": "ok",
                "config": newConfig.to_dict(),
                "warnings": warnings,
            })
        except (ValueError, KeyError) as e:
            self._sendJSON(400, {"error": str(e)})
        except Exception as e:
            self._sendJSON(500, {"error": f"server error: {e}"})

    def _handleRebuildVocab(self):
        """POST /api/tokenizer/rebuild — rebuild vocabulary with current config."""
        try:
            from tokenizer import (
                WordTokenizer, TokenizerConfig,
                loadClaudeOpusDataset, loadWikipediaDataset,
                extractTextsForVocab
            )

            # Load current config from vocab.json
            config_dict = getVocabConfig()
            if config_dict:
                config = TokenizerConfig.from_dict(config_dict)
            else:
                config = TokenizerConfig()

            # Load data based on config
            if config.dataset_source == "claude-opus":
                texts = loadClaudeOpusDataset(
                    datasetName=config.dataset_name,
                    numRows=config.dataset_rows,
                )
            else:
                texts = loadWikipediaDataset(numArticles=config.dataset_rows)

            # Build and save
            tokenizer = WordTokenizer(config=config)
            tokenizer.buildVocab(extractTextsForVocab(texts))
            tokenizer.save(VOCAB_PATH)

            # Reload in-memory state
            reloadVocab()

            self._sendJSON(200, {
                "status": "ok",
                "vocabSize": tokenizer.vocabSize,
                "config": tokenizer.get_config(),
            })
        except Exception as e:
            self._sendJSON(500, {"error": f"rebuild failed: {e}"})

    # ── Utilities ─────────────────────────────────────────────────────────

    def _sendJSON(self, code, obj):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(obj).encode())

    def do_OPTIONS(self):
        """Handle CORS preflight for POST requests."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        pass


def main():
    print(f"{'=' * 50}")
    print(f"  📊 TRANSFORMER DASHBOARD")
    print(f"{'=' * 50}")
    print(f"  Serving on http://localhost:{PORT}")
    print(f"  Metrics:  {METRICS_PATH}")
    print(f"  Vocab:    {VOCAB_PATH}")
    print(f"  Press Ctrl+C to stop.\n")

    server = HTTPServer(("", PORT), DashboardHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Dashboard stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
