"""Small persisted online logistic scorer for Hive shadow calibration.

The scorer is deliberately conservative. It learns transport reliability from
future route outcomes, but it never selects a model by itself. Hive and Jev
remain the active policy until this candidate has enough evidence to be
evaluated against the current policy.
"""

import json
import math
import os
import threading
import time


STATE = os.path.expanduser("~/.codex/codex-router")
MODEL_PATH = os.path.join(STATE, "hive-logistic.json")
MODEL_VERSION = 1
LEARNING_RATE = 0.05
L2 = 0.001
MIN_EXAMPLES = 200
_lock = threading.Lock()


def _default_model():
    return {
        "version": MODEL_VERSION,
        "enabled": True,
        "mode": "shadow",
        "labelSource": "transport_status",
        "learningRate": LEARNING_RATE,
        "l2": L2,
        "minExamples": MIN_EXAMPLES,
        "examples": 0,
        "positive": 0,
        "negative": 0,
        "bias": 0.0,
        "weights": {},
        "groups": {},
        "logLoss": 0.0,
        "lastUpdate": 0.0,
    }


def _load():
    model = _default_model()
    try:
        with open(MODEL_PATH, encoding="utf-8") as handle:
            saved = json.load(handle)
        if isinstance(saved, dict) and saved.get("version") == MODEL_VERSION:
            for key in model:
                if key in saved:
                    model[key] = saved[key]
    except (OSError, ValueError, TypeError):
        pass
    return model


def _save(model):
    os.makedirs(STATE, mode=0o700, exist_ok=True)
    temporary = MODEL_PATH + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(model, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, MODEL_PATH)


def _number(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _features(features):
    """Convert bounded metadata into a sparse, deterministic feature vector."""
    features = features if isinstance(features, dict) else {}
    vector = {
        "bias": 1.0,
        "confidence": max(0.0, min(1.0, _number(features.get("confidence"), 0.5))),
        "context_chars_log": min(20.0, math.log1p(max(0.0, _number(features.get("context_chars"))))),
        "context_items_log": min(12.0, math.log1p(max(0.0, _number(features.get("context_items"))))),
        "errored": 1.0 if features.get("errored") else 0.0,
        "deepseek_replay": 1.0 if features.get("deepseek_replay") else 0.0,
        "phase_cacheable": 1.0 if features.get("phase_cacheable") else 0.0,
    }
    for key in ("route_class", "model", "effort", "step_type", "reason"):
        value = features.get(key)
        if isinstance(value, str) and value:
            vector[f"{key}={value}"] = 1.0
    route_class = features.get("route_class")
    model = features.get("model")
    if isinstance(route_class, str) and isinstance(model, str):
        vector[f"route_class:model={route_class}|{model}"] = 1.0
    return vector


def _sigmoid(value):
    value = max(-35.0, min(35.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def predict(features):
    """Return a shadow probability without mutating the persisted model."""
    with _lock:
        model = _load()
    vector = _features(features)
    score = _number(model.get("bias"))
    for name, value in vector.items():
        if name != "bias":
            score += _number(model.get("weights", {}).get(name)) * value
    return round(_sigmoid(score), 6)


def update(features, status):
    """Apply one transport outcome and persist the updated candidate model."""
    label = 1 if int(status) == 200 else 0
    vector = _features(features)
    with _lock:
        model = _load()
        weights = model.setdefault("weights", {})
        score = _number(model.get("bias"))
        for name, value in vector.items():
            if name != "bias":
                score += _number(weights.get(name)) * value
        probability = _sigmoid(score)
        learning_rate = _number(model.get("learningRate"), LEARNING_RATE)
        regularization = _number(model.get("l2"), L2)
        error = label - probability
        model["bias"] = _number(model.get("bias")) + learning_rate * error
        for name, value in vector.items():
            if name == "bias":
                continue
            weights[name] = (
                _number(weights.get(name))
                + learning_rate * (error * value - regularization * _number(weights.get(name)))
            )
        model["examples"] = int(model.get("examples", 0) or 0) + 1
        model["positive" if label else "negative"] = int(
            model.get("positive" if label else "negative", 0) or 0
        ) + 1
        group = f"{features.get('route_class', 'other')}|{features.get('model', 'unknown')}"
        groups = model.setdefault("groups", {})
        groups[group] = int(groups.get(group, 0) or 0) + 1
        old_loss = _number(model.get("logLoss"))
        examples = int(model["examples"])
        clipped = max(1e-6, min(1.0 - 1e-6, probability))
        model["logLoss"] = old_loss + ((-math.log(clipped) if label else -math.log(1.0 - clipped)) - old_loss) / examples
        model["lastUpdate"] = time.time()
        _save(model)
        return {
            "probability": round(probability, 6),
            "label": label,
            "examples": examples,
        }


def snapshot():
    with _lock:
        model = _load()
    examples = int(model.get("examples", 0) or 0)
    positive = int(model.get("positive", 0) or 0)
    negative = int(model.get("negative", 0) or 0)
    return {
        "enabled": bool(model.get("enabled", True)),
        "mode": model.get("mode", "shadow"),
        "labelSource": model.get("labelSource", "transport_status"),
        "examples": examples,
        "positive": positive,
        "negative": negative,
        "baseRate": round(positive / examples, 4) if examples else 0.0,
        "logLoss": round(_number(model.get("logLoss")), 6),
        "minExamples": int(model.get("minExamples", MIN_EXAMPLES) or MIN_EXAMPLES),
        "candidateReady": examples >= int(model.get("minExamples", MIN_EXAMPLES) or MIN_EXAMPLES),
        "lastUpdate": _number(model.get("lastUpdate")),
        "groupCounts": dict(model.get("groups") or {}),
        "featureCount": len(model.get("weights") or {}),
    }
