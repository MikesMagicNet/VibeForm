# autotuner.py — Active Configuration Matcher
#
# Monitors live training metrics and dynamically adjusts hyperparameters.
# Runs inline during training — zero overhead, no extra processes.
#
# What it watches:
#   • Loss plateau detection (stalled learning → LR boost or schedule reset)
#   • Gradient health (exploding/vanishing → adjust clip norm + LR)
#   • Accuracy trend analysis (stalling → adjust label smoothing)
#   • Overconfidence detection (high acc + flat loss → increase label smoothing)
#   • Perplexity spikes (divergence detection → emergency LR cut)
#
# How to use:
#   from autotuner import ActiveConfigMatcher
#   tuner = ActiveConfigMatcher(config, metrics)
#   # Inside training loop:
#   tuner.step(batchIdx, loss, gradNorm, accuracy, lr)

import math
import time
from collections import deque


class ActiveConfigMatcher:
    """
    Watches training telemetry in real-time and adjusts hyperparameters
    to keep training healthy and progressing.

    Design principles:
      - Conservative: small adjustments, never wild swings
      - Logged: every adjustment is recorded for dashboard display
      - Cooldown: waits N steps between adjustments to measure effect
      - Reversible: tracks what it changed so you can audit
    """

    def __init__(self, config, metrics, enabled=True):
        self.config = config
        self.metrics = metrics
        self.enabled = enabled

        # ── Windows for trend analysis ────────────────────────────────────
        self.windowSize = 100         # steps to analyze for trends
        self.lossWindow = deque(maxlen=self.windowSize)
        self.gradWindow = deque(maxlen=self.windowSize)
        self.accWindow = deque(maxlen=self.windowSize)
        self.lrHistory = deque(maxlen=self.windowSize)

        # ── Cooldowns (prevent rapid-fire adjustments) ────────────────────
        self.cooldownSteps = 200      # minimum steps between adjustments
        self.lastAdjustStep = -self.cooldownSteps  # allow first adjustment immediately
        self.stepCount = 0

        # ── Thresholds ────────────────────────────────────────────────────
        # Gradient health
        self.gradExplosionThreshold = 10.0     # grad norm above this = exploding
        self.gradVanishingThreshold = 1e-4     # grad norm below this = vanishing
        self.gradHighFrequency = 0.3           # 30% of window above explosion = problem

        # Loss plateau detection
        self.plateauThreshold = 0.002          # relative improvement < 0.2% = plateau
        self.plateauWindow = 50                # check plateau over this many steps

        # Accuracy stall detection
        self.accStallThreshold = 0.001         # absolute improvement < 0.1% = stall
        self.accStallWindow = 80               # steps to detect accuracy stall

        # Overconfidence detection
        self.confidenceWindow = deque(maxlen=self.windowSize)  # (accuracy, loss) pairs
        self.overconfidenceAccThreshold = 0.85  # accuracy above this triggers scrutiny
        self.overconfidenceLossRatio = 0.005    # if loss improves < 0.5% but acc > threshold
        self.accLossDivergenceWindow = 60       # steps to check acc↑ vs loss↑ divergence

        # Divergence detection
        self.divergenceMultiplier = 3.0        # loss spike > 3x recent avg = divergence
        self.nanCount = 0

        # ── Adjustment limits (safety rails) ──────────────────────────────
        self.minLR = config.get("minLR", 1e-6) * 0.1       # absolute floor
        self.maxLR = config.get("maxLR", 5e-4) * 3.0       # absolute ceiling
        self.minGradClip = 0.1
        self.maxGradClip = 10.0
        self.minLabelSmoothing = 0.0
        self.maxLabelSmoothing = 0.3
        self.minDropout = 0.0
        self.maxDropout = 0.5

        # ── State tracking ────────────────────────────────────────────────
        self.adjustments = []          # log of all adjustments made
        self.healthStatus = "healthy"  # healthy, warning, critical
        self.healthReasons = []

        # ── Current effective config (starts from initial config) ─────────
        self.effectiveLR = config.get("maxLR", 5e-4)
        self.effectiveGradClip = config.get("gradClipNorm", 1.1)
        self.effectiveLabelSmoothing = config.get("labelSmoothing", 0.2)
        self.effectiveDropout = config.get("dropout", 0.1)
        self.lrScale = 1.0            # multiplier applied to the scheduled LR
        self.archRecommendations = [] # un-tunable architecture fixes

        # Initialize metrics tracking
        self._updateMetrics()

    # =====================================================================
    #  MAIN STEP — Called every training batch
    # =====================================================================

    def step(self, batchIdx, loss, gradNorm, accuracy, currentLR, metrics_data=None):
        """
        Analyze current metrics and apply adjustments if needed.

        Returns:
            dict with any adjusted values:
              {"maxLR": ..., "gradClipNorm": ..., "labelSmoothing": ..., "lrScale": ...}
            Empty dict if no adjustments were made.
        """
        if not self.enabled:
            return {}

        self.stepCount += 1

        # Record metrics
        if not math.isnan(loss) and not math.isinf(loss):
            self.lossWindow.append(loss)
        else:
            self.nanCount += 1

        if not math.isnan(gradNorm):
            self.gradWindow.append(gradNorm)

        self.accWindow.append(accuracy)
        self.lrHistory.append(currentLR)
        self.confidenceWindow.append((accuracy, loss if not math.isnan(loss) else 0.0))

        # ── Run diagnostic checks ────────────────────────────────────────
        self.healthReasons = []
        adjustments = {}

        # Only analyze after enough data is collected
        if self.stepCount < 30:
            self.healthStatus = "warmup"
            self._updateMetrics()
            return {}

        # Check each health dimension
        gradAdj = self._checkGradientHealth()
        lossAdj = self._checkLossPlateau()
        accAdj = self._checkAccuracyTrend()
        confAdj = self._checkOverconfidence()
        divAdj = self._checkDivergence(loss)
        sftAdj = self._checkSelfFineTune(metrics_data, loss, accuracy)

        # Merge adjustments (priority: divergence > sft > gradients > overconfidence > plateau > accuracy)
        for adj in [accAdj, lossAdj, confAdj, gradAdj, sftAdj, divAdj]:
            adjustments.update(adj)

        # Apply cooldown check
        if adjustments and (self.stepCount - self.lastAdjustStep) < self.cooldownSteps:
            # Only allow emergency (divergence) adjustments during cooldown
            if "emergency" not in adjustments:
                adjustments = {}

        # Apply adjustments
        if adjustments:
            adjustments.pop("emergency", None)  # remove internal flag
            self._applyAdjustments(adjustments)
            self.lastAdjustStep = self.stepCount

        # Determine overall health status
        if self.nanCount > 5 or "critical" in str(self.healthReasons):
            self.healthStatus = "critical"
        elif self.healthReasons:
            self.healthStatus = "warning"
        else:
            self.healthStatus = "healthy"

        self._updateMetrics()
        return adjustments

    # =====================================================================
    #  GRADIENT HEALTH CHECK
    # =====================================================================

    def _checkGradientHealth(self):
        if len(self.gradWindow) < 20:
            return {}

        grads = list(self.gradWindow)
        recentGrads = grads[-20:]
        avgGrad = sum(recentGrads) / len(recentGrads)
        maxGrad = max(recentGrads)

        adjustments = {}

        # ── Exploding gradients ───────────────────────────────────────────
        explosionCount = sum(1 for g in recentGrads if g > self.gradExplosionThreshold)
        explosionRate = explosionCount / len(recentGrads)

        if explosionRate > self.gradHighFrequency:
            self.healthReasons.append("gradient_explosion")
            # Tighten gradient clipping
            newClip = max(self.effectiveGradClip * 0.8, self.minGradClip)
            if newClip != self.effectiveGradClip:
                adjustments["gradClipNorm"] = round(newClip, 4)
            # Also reduce LR scale
            newScale = max(self.lrScale * 0.85, 0.1)
            if newScale != self.lrScale:
                adjustments["lrScale"] = round(newScale, 4)

        elif maxGrad > self.gradExplosionThreshold * 2:
            self.healthReasons.append("gradient_spike")
            newClip = max(self.effectiveGradClip * 0.9, self.minGradClip)
            adjustments["gradClipNorm"] = round(newClip, 4)

        # ── Vanishing gradients ───────────────────────────────────────────
        vanishCount = sum(1 for g in recentGrads if g < self.gradVanishingThreshold)
        vanishRate = vanishCount / len(recentGrads)

        if vanishRate > 0.5:
            self.healthReasons.append("gradient_vanishing")
            # Loosen gradient clipping
            newClip = min(self.effectiveGradClip * 1.2, self.maxGradClip)
            if newClip != self.effectiveGradClip:
                adjustments["gradClipNorm"] = round(newClip, 4)
            # Boost LR
            newScale = min(self.lrScale * 1.15, 3.0)
            if newScale != self.lrScale:
                adjustments["lrScale"] = round(newScale, 4)

        return adjustments

    # =====================================================================
    #  LOSS PLATEAU DETECTION
    # =====================================================================

    def _checkLossPlateau(self):
        if len(self.lossWindow) < self.plateauWindow:
            return {}

        losses = list(self.lossWindow)
        halfPoint = len(losses) // 2
        firstHalf = losses[:halfPoint]
        secondHalf = losses[halfPoint:]

        avgFirst = sum(firstHalf) / len(firstHalf)
        avgSecond = sum(secondHalf) / len(secondHalf)

        # Relative improvement
        if avgFirst > 0:
            improvement = (avgFirst - avgSecond) / avgFirst
        else:
            improvement = 0

        if improvement < self.plateauThreshold and improvement >= 0:
            self.healthReasons.append("loss_plateau")
            # Boost LR to escape plateau
            newScale = min(self.lrScale * 1.1, 3.0)
            return {"lrScale": round(newScale, 4)} if newScale != self.lrScale else {}

        return {}

    # =====================================================================
    #  ACCURACY TREND ANALYSIS
    # =====================================================================

    def _checkAccuracyTrend(self):
        if len(self.accWindow) < self.accStallWindow:
            return {}

        accs = list(self.accWindow)
        halfPoint = len(accs) // 2
        firstHalf = accs[:halfPoint]
        secondHalf = accs[halfPoint:]

        avgFirst = sum(firstHalf) / len(firstHalf)
        avgSecond = sum(secondHalf) / len(secondHalf)

        improvement = avgSecond - avgFirst

        if improvement < self.accStallThreshold and improvement >= 0:
            self.healthReasons.append("accuracy_stall")
            # Reduce label smoothing slightly to sharpen predictions
            newSmoothing = max(self.effectiveLabelSmoothing - 0.02, self.minLabelSmoothing)
            if newSmoothing != self.effectiveLabelSmoothing:
                return {"labelSmoothing": round(newSmoothing, 4)}

        elif improvement < -0.01:
            # Accuracy is actively degrading
            self.healthReasons.append("accuracy_degrading")
            newScale = max(self.lrScale * 0.9, 0.1)
            return {"lrScale": round(newScale, 4)} if newScale != self.lrScale else {}

        return {}

    # =====================================================================
    #  OVERCONFIDENCE DETECTION
    #  Detects when model is too confident (high acc but loss not improving)
    #  and increases label smoothing to regularize.
    # =====================================================================

    def _checkOverconfidence(self):
        if len(self.confidenceWindow) < self.accLossDivergenceWindow:
            return {}

        pairs = list(self.confidenceWindow)
        recentPairs = pairs[-self.accLossDivergenceWindow:]
        halfPoint = len(recentPairs) // 2
        firstHalf = recentPairs[:halfPoint]
        secondHalf = recentPairs[halfPoint:]

        avgAccFirst = sum(a for a, _ in firstHalf) / len(firstHalf)
        avgAccSecond = sum(a for a, _ in secondHalf) / len(secondHalf)
        avgLossFirst = sum(l for _, l in firstHalf) / len(firstHalf)
        avgLossSecond = sum(l for _, l in secondHalf) / len(secondHalf)

        accImproving = avgAccSecond > avgAccFirst + 0.005  # accuracy rising

        # Pattern 1: High accuracy but loss is flat or increasing
        # The model is memorizing — predictions are confident but wrong patterns
        if avgLossFirst > 0:
            lossImprovement = (avgLossFirst - avgLossSecond) / avgLossFirst
        else:
            lossImprovement = 0

        lossStagnant = lossImprovement < self.overconfidenceLossRatio

        if avgAccSecond > self.overconfidenceAccThreshold and lossStagnant:
            self.healthReasons.append("overconfidence_high_acc")
            newSmoothing = min(
                self.effectiveLabelSmoothing + 0.025,
                self.maxLabelSmoothing
            )
            if newSmoothing != self.effectiveLabelSmoothing:
                return {"labelSmoothing": round(newSmoothing, 4)}

        # Pattern 2: Accuracy climbing while loss is also climbing
        # Classic overfit divergence — model is getting more confident on wrong things
        lossIncreasing = avgLossSecond > avgLossFirst * 1.01  # loss rose > 1%

        if accImproving and lossIncreasing:
            self.healthReasons.append("overconfidence_divergence")
            newSmoothing = min(
                self.effectiveLabelSmoothing + 0.03,
                self.maxLabelSmoothing
            )
            if newSmoothing != self.effectiveLabelSmoothing:
                return {"labelSmoothing": round(newSmoothing, 4)}

        # Pattern 3: Suspiciously high accuracy early in training
        # If acc > 90% in the first 500 steps, something is off
        if self.stepCount < 500 and avgAccSecond > 0.90:
            self.healthReasons.append("overconfidence_early")
            newSmoothing = min(
                self.effectiveLabelSmoothing + 0.04,
                self.maxLabelSmoothing
            )
            if newSmoothing != self.effectiveLabelSmoothing:
                return {"labelSmoothing": round(newSmoothing, 4)}

        return {}

    # =====================================================================
    #  DIVERGENCE DETECTION
    # =====================================================================

    def _checkDivergence(self, currentLoss):
        if len(self.lossWindow) < 10:
            return {}

        # Check for NaN/Inf
        if math.isnan(currentLoss) or math.isinf(currentLoss):
            self.healthReasons.append("nan_loss_critical")
            return {
                "lrScale": round(max(self.lrScale * 0.5, 0.05), 4),
                "gradClipNorm": round(max(self.effectiveGradClip * 0.5, self.minGradClip), 4),
                "emergency": True,
            }

        # Check for loss spikes
        recentLosses = list(self.lossWindow)[-20:]
        avgRecent = sum(recentLosses) / len(recentLosses)

        if currentLoss > avgRecent * self.divergenceMultiplier and avgRecent > 0:
            self.healthReasons.append("loss_spike")
            return {
                "lrScale": round(max(self.lrScale * 0.7, 0.1), 4),
                "emergency": True,
            }

        return {}

    # =====================================================================
    #  APPLY ADJUSTMENTS
    # =====================================================================

    def _applyAdjustments(self, adjustments):
        """Apply adjustments and log them."""
        record = {
            "step": self.stepCount,
            "timestamp": time.time(),
            "reasons": list(self.healthReasons),
            "changes": {},
            "before": {},
        }

        if "gradClipNorm" in adjustments:
            record["before"]["gradClipNorm"] = self.effectiveGradClip
            self.effectiveGradClip = adjustments["gradClipNorm"]
            record["changes"]["gradClipNorm"] = self.effectiveGradClip

        if "lrScale" in adjustments:
            record["before"]["lrScale"] = self.lrScale
            self.lrScale = adjustments["lrScale"]
            record["changes"]["lrScale"] = self.lrScale

        if "labelSmoothing" in adjustments:
            record["before"]["labelSmoothing"] = self.effectiveLabelSmoothing
            self.effectiveLabelSmoothing = adjustments["labelSmoothing"]
            record["changes"]["labelSmoothing"] = self.effectiveLabelSmoothing

        if "dropout" in adjustments:
            record["before"]["dropout"] = self.effectiveDropout
            self.effectiveDropout = adjustments["dropout"]
            record["changes"]["dropout"] = self.effectiveDropout

        self.adjustments.append(record)

        # Console output
        reasons = ", ".join(self.healthReasons)
        changes = ", ".join(f"{k}={v}" for k, v in record["changes"].items())
        print(f"\n  🔧 AutoTuner [{self.healthStatus.upper()}] step {self.stepCount}: "
              f"{reasons}")
        print(f"     Adjusted: {changes}")

    # =====================================================================
    #  SCALED LEARNING RATE — Apply to scheduled LR
    # =====================================================================

    def scaleLR(self, scheduledLR):
        """
        Apply the tuner's LR scale factor to the scheduled learning rate.
        Call this after getLearningRate() to get the effective LR.
        """
        return scheduledLR * self.lrScale

    # =====================================================================
    #  OVERCONFIDENCE RISK SCORE — 0.0 (healthy) to 1.0 (overconfident)
    # =====================================================================

    def _getOverconfidenceRisk(self):
        """Compute a 0-1 risk score for overconfidence."""
        if len(self.confidenceWindow) < 30:
            return 0.0

        pairs = list(self.confidenceWindow)[-30:]
        avgAcc = sum(a for a, _ in pairs) / len(pairs)
        avgLoss = sum(l for _, l in pairs) / len(pairs)

        # Risk increases as accuracy goes up without loss going down
        accRisk = max(0, (avgAcc - 0.5)) / 0.5   # 0 at 50% acc, 1 at 100%

        # Check if loss is improving (lower = better)
        if len(self.lossWindow) >= 30:
            losses = list(self.lossWindow)
            earlyLoss = sum(losses[:15]) / 15
            lateLoss = sum(losses[-15:]) / 15
            if earlyLoss > 0:
                lossProgress = max(0, 1 - (earlyLoss - lateLoss) / earlyLoss)
            else:
                lossProgress = 0
        else:
            lossProgress = 0

        # Combine: high acc + no loss progress = high risk
        risk = accRisk * lossProgress
        return round(min(risk, 1.0), 4)

    # =====================================================================
    #  SELF-FINE-TUNE INTEGRATION (Live SFT + Architect Recommendations)
    # =====================================================================

    def _checkSelfFineTune(self, metrics_data, currentLoss, currentAcc):
        """Reads extra telemetry (val loss, entropy, memory) and provides recommendations or live adjustments."""
        adjustments = {}
        self.archRecommendations = []
        if not metrics_data:
            return adjustments

        # 1. Overfitting Check: Validation loss > Train loss * 1.15
        valLosses = metrics_data.get("valLosses", [])
        if valLosses and len(valLosses) > 0:
            vl = valLosses[-1]
            if vl > currentLoss * 1.15 and currentAcc > 0.4:
                self.healthReasons.append("sft_overfitting")
                newDropout = min(self.effectiveDropout + 0.05, self.maxDropout)
                if newDropout != self.effectiveDropout:
                    adjustments["dropout"] = round(newDropout, 4)

        # 2. Capacity Check (Requires restart)
        epochs = metrics_data.get("currentEpoch", 1)
        if currentLoss > 4.5 and epochs > 2:
            self.archRecommendations.append({"param": "dModel / Heads", "action": "Increase", "reason": "High training loss after multiple epochs. Model capacity may be too low."})

        # 3. Attention Entropy
        ent = metrics_data.get("attnEntropy", 0)
        if ent > 0 and ent < 1.5:
            self.archRecommendations.append({"param": "Heads (h)", "action": "Increase", "reason": f"Low attention entropy ({ent:.2f}). Attention is too focused."})
        elif ent > 5.0:
            self.archRecommendations.append({"param": "Layers (N)", "action": "Decrease", "reason": f"High attention entropy ({ent:.2f}). Attention is too diffuse."})

        # 4. Memory
        mem = metrics_data.get("memoryUsage", 0)
        if mem > 14.0:
            self.archRecommendations.append({"param": "Batch Size / Seq Len", "action": "Decrease", "reason": f"Memory usage is very high ({mem:.1f} GB)."})

        return adjustments

    # =====================================================================
    #  UPDATE METRICS — Feed state to dashboard
    # =====================================================================

    def _updateMetrics(self):
        """Push tuner state into metrics for dashboard visualization."""
        # Compute current diagnostics
        diagnostics = {
            "avgLoss": round(sum(self.lossWindow) / max(len(self.lossWindow), 1), 4),
            "avgGradNorm": round(sum(self.gradWindow) / max(len(self.gradWindow), 1), 4),
            "avgAccuracy": round(sum(self.accWindow) / max(len(self.accWindow), 1), 4),
            "maxGradNorm": round(max(self.gradWindow) if self.gradWindow else 0, 4),
            "minGradNorm": round(min(self.gradWindow) if self.gradWindow else 0, 6),
            "overconfidenceRisk": self._getOverconfidenceRisk(),
        }

        tunerState = {
            "enabled": self.enabled,
            "healthStatus": self.healthStatus,
            "healthReasons": list(self.healthReasons),
            "stepCount": self.stepCount,
            "lrScale": round(self.lrScale, 4),
            "effectiveGradClip": round(self.effectiveGradClip, 4),
            "effectiveLabelSmoothing": round(self.effectiveLabelSmoothing, 4),
            "effectiveDropout": round(self.effectiveDropout, 4),
            "totalAdjustments": len(self.adjustments),
            "recentAdjustments": self.adjustments[-10:],  # last 10 for dashboard
            "archRecommendations": self.archRecommendations,
            "diagnostics": diagnostics,
            "nanCount": self.nanCount,
            "cooldownRemaining": max(0, self.cooldownSteps - (self.stepCount - self.lastAdjustStep)),
        }

        self.metrics.update(autotuner=tunerState)

    # =====================================================================
    #  SUMMARY — End of training report
    # =====================================================================

    def summary(self):
        """Print a summary of all auto-tuning activity."""
        print(f"\n{'═' * 60}")
        print(f"  🔧 AUTO-TUNER SUMMARY")
        print(f"{'═' * 60}")
        print(f"  Total steps monitored:    {self.stepCount:,}")
        print(f"  Total adjustments:        {len(self.adjustments)}")
        print(f"  Final health status:      {self.healthStatus}")
        print(f"  Final LR scale:           {self.lrScale:.4f}")
        print(f"  Final grad clip:          {self.effectiveGradClip:.4f}")
        print(f"  Final label smoothing:    {self.effectiveLabelSmoothing:.4f}")
        print(f"  NaN events:               {self.nanCount}")

        if self.adjustments:
            print(f"\n  ── Adjustment Log ──")
            for adj in self.adjustments:
                step = adj["step"]
                reasons = ", ".join(adj["reasons"])
                changes = ", ".join(f"{k}={v}" for k, v in adj["changes"].items())
                print(f"     Step {step:>6}: [{reasons}] → {changes}")

        print(f"{'═' * 60}\n")
