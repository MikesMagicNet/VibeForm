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
from config import PATHS, DASHBOARD

METRICS_PATH = PATHS["metrics"]
VOCAB_PATH = PATHS["vocab"]
PORT = DASHBOARD["port"]

# Load vocabulary once at startup for the token visualizer API
_vocab = None
def getVocab():
    global _vocab
    if _vocab is None and os.path.exists(VOCAB_PATH):
        with open(VOCAB_PATH, "r") as f:
            data = json.load(f)
        _vocab = {
            "word2id": data["word2id"],
            "id2word": {int(k): v for k, v in data["id2word"].items()},
            "vocabSize": data["vocabSize"],
        }
    return _vocab


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
        return super().do_GET()

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
        from urllib.parse import urlparse, parse_qs
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

    def _sendJSON(self, code, obj):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(obj).encode())

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
