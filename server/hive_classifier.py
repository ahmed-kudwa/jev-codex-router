"""Cheap local routing decisions with bounded adaptive feedback.

The classifier intentionally handles only high-confidence cases. Ambiguous
requests return None and remain Jev's responsibility. Feedback is a small
owner-local ledger: successful responses strengthen a route, provider errors
weaken it, and repeated failures temporarily quarantine that route.
"""

import json
import hashlib
import os
import re
import threading
import time
from datetime import datetime

from hive_logistic import predict as logistic_predict
from hive_logistic import snapshot as logistic_snapshot
from hive_logistic import update as logistic_update


STATE = os.path.expanduser("~/.codex/codex-router")
POLICY_PATH = os.path.join(STATE, "hive-policy.json")
EVENTS_PATH = os.path.join(STATE, "hive-events.jsonl")
CACHE_PATH = os.path.join(STATE, "hive-jev-cache.json")
CACHE_MAINTENANCE_PATH = os.path.join(STATE, "hive-maintenance.json")
CACHE_TTL_SECONDS = 600
CACHE_MAX_ENTRIES = 256
MAINTENANCE_INTERVAL_SECONDS = 3600
ADAPTIVE_WINDOW_SECONDS = 30 * 24 * 3600
ADAPTIVE_MIN_ATTEMPTS = 8
ADAPTIVE_MIN_SUCCESS_RATE = 0.90
ADAPTIVE_MAX_EVENTS = 20000
_cache_lock = threading.Lock()
_policy_lock = threading.Lock()

LUNA = "gpt-5.6-luna"
SOL = "gpt-5.6-sol"
ASTRA = "gpt-6-astra"
DEEPSEEK = "opencode-go/deepseek-v4.1-flash"
MODEL_FAILURE_LIMIT = 0.25
MODEL_ORDER = (DEEPSEEK, LUNA, SOL, ASTRA)

CRITICAL_RE = re.compile(
    r"(?i)\b(security|auth(?:entication|orization)?|production|deploy|release|"
    r"financial|payment|destructive|delete|privacy|incident|outage)\b"
)
HARD_RE = re.compile(
    r"(?i)\b(architecture|migration|schema|race condition|concurrency|"
    r"message bus|database queue|distributed)\b"
)
MECHANICAL_RE = re.compile(
    r"(?i)\b(read|list|show|inspect|check|run|rerun|format|lint|typecheck|"
    r"rename|move|apply|patch|grep|search|find|verify|compare|status|summarize)\b"
)
REVIEW_RE = re.compile(r"(?i)\b(review|audit|regression|code review|final verification)\b")
FILE_RE = re.compile(r"(?i)(?:[\w./-]+\.(?:py|js|jsx|ts|tsx|go|rs|java|sql|yaml|yml|json|toml|md))\b")


def _default_policy():
    return {
        "version": 1,
        "enabled": True,
        "minAttemptsForQuarantine": 6,
        "failureRateForQuarantine": 0.35,
        "quarantineSeconds": 900,
        "models": {
            DEEPSEEK: {"success": 0, "failure": 0, "quarantinedUntil": 0},
            LUNA: {"success": 0, "failure": 0, "quarantinedUntil": 0},
            SOL: {"success": 0, "failure": 0, "quarantinedUntil": 0},
            ASTRA: {"success": 0, "failure": 0, "quarantinedUntil": 0},
        },
        "adaptive": {
            "updatedAt": 0,
            "windowSeconds": ADAPTIVE_WINDOW_SECONDS,
            "minAttempts": ADAPTIVE_MIN_ATTEMPTS,
            "minSuccessRate": ADAPTIVE_MIN_SUCCESS_RATE,
            "eventsConsidered": 0,
            "lastEventAt": 0,
            "models": {},
            "classes": {},
            "recommendations": {},
        },
    }


def _load_policy():
    policy = _default_policy()
    try:
        with open(POLICY_PATH, encoding="utf-8") as handle:
            saved = json.load(handle)
        if isinstance(saved, dict) and saved.get("version") == 1:
            policy.update({key: saved[key] for key in ("enabled", "minAttemptsForQuarantine",
                                                        "failureRateForQuarantine",
                                                        "quarantineSeconds")
                           if key in saved})
            for model, stats in (saved.get("models") or {}).items():
                if isinstance(stats, dict):
                    policy["models"].setdefault(model, {}).update({
                        key: int(stats.get(key, 0) or 0)
                        for key in ("success", "failure", "quarantinedUntil")
                    })
            adaptive = saved.get("adaptive")
            if isinstance(adaptive, dict):
                current = policy["adaptive"]
                for key in ("updatedAt", "windowSeconds", "minAttempts", "eventsConsidered", "lastEventAt"):
                    if key in adaptive:
                        current[key] = int(adaptive[key] or 0)
                if "minSuccessRate" in adaptive:
                    current["minSuccessRate"] = float(adaptive["minSuccessRate"] or 0)
                for key in ("models", "classes", "recommendations"):
                    if isinstance(adaptive.get(key), dict):
                        current[key] = adaptive[key]
    except (OSError, ValueError, TypeError):
        pass
    return policy


def _save_policy(policy):
    os.makedirs(STATE, mode=0o700, exist_ok=True)
    temporary = POLICY_PATH + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(policy, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, POLICY_PATH)


def _load_cache():
    try:
        with open(CACHE_PATH, encoding="utf-8") as handle:
            saved = json.load(handle)
        if isinstance(saved, dict) and saved.get("version") == 1:
            saved.setdefault("entries", {})
            saved.setdefault("stats", {"hits": 0, "misses": 0, "evictions": 0})
            return saved
    except (OSError, ValueError, TypeError):
        pass
    return {"version": 1, "entries": {}, "stats": {"hits": 0, "misses": 0, "evictions": 0}}


def _save_cache(cache):
    os.makedirs(STATE, mode=0o700, exist_ok=True)
    temporary = CACHE_PATH + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(cache, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, CACHE_PATH)


def cache_get(key, catalog):
    """Return a still-valid Jev route cache entry, or None.

    Only the hashed key and the already-routed model are persisted; prompts and
    tool output never enter this cache.
    """
    if not key:
        return None
    now = time.time()
    with _cache_lock:
        cache = _load_cache()
        entry = cache.get("entries", {}).get(key)
        if not isinstance(entry, dict) or now - float(entry.get("at", 0) or 0) > CACHE_TTL_SECONDS:
            cache["stats"]["misses"] = int(cache["stats"].get("misses", 0)) + 1
            _save_cache(cache)
            return None
        route = entry.get("route")
        if not isinstance(route, dict) or route.get("model") not in catalog:
            cache.get("entries", {}).pop(key, None)
            cache["stats"]["misses"] = int(cache["stats"].get("misses", 0)) + 1
            _save_cache(cache)
            return None
        cache["stats"]["hits"] = int(cache["stats"].get("hits", 0)) + 1
        _save_cache(cache)
        return dict(route)


def cache_put(key, route):
    if not key or not isinstance(route, dict) or not route.get("model"):
        return
    with _cache_lock:
        cache = _load_cache()
        entries = cache.setdefault("entries", {})
        entries[key] = {"at": time.time(), "route": route}
        if len(entries) > CACHE_MAX_ENTRIES:
            oldest = sorted(entries.items(), key=lambda item: float(item[1].get("at", 0) or 0))
            for old_key, _ in oldest[: len(entries) - CACHE_MAX_ENTRIES]:
                entries.pop(old_key, None)
                cache["stats"]["evictions"] = int(cache["stats"].get("evictions", 0)) + 1
        _save_cache(cache)


def cache_invalidate(key):
    if not key:
        return
    with _cache_lock:
        cache = _load_cache()
        if cache.get("entries", {}).pop(key, None) is not None:
            _save_cache(cache)


def cache_key(session_key, task, step):
    """Build a stable, private phase key without persisting prompt text."""
    if not session_key or not task:
        return None
    normalized = re.sub(r"\s+", " ", task.strip().lower())[:500]
    phase = step.get("step_type") if isinstance(step, dict) else "other"
    material = json.dumps({"session": session_key, "phase": phase, "task": normalized}, sort_keys=True)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


def cache_snapshot():
    with _cache_lock:
        cache = _load_cache()
    return {
        "entries": len(cache.get("entries", {})),
        "ttlSeconds": CACHE_TTL_SECONDS,
        "maxEntries": CACHE_MAX_ENTRIES,
        "hits": int(cache.get("stats", {}).get("hits", 0)),
        "misses": int(cache.get("stats", {}).get("misses", 0)),
        "evictions": int(cache.get("stats", {}).get("evictions", 0)),
    }


def maintenance(force=False):
    """Persist hourly health maintenance across router restarts.

    Route health is updated on every outcome. This job also recalibrates the
    durable adaptive policy from the event ledger, so closing Codex cannot
    erase learned state and reopening the laptop refreshes it immediately.
    """
    now = time.time()
    with _cache_lock:
        state = _load_cache()
        previous = _load_json(CACHE_MAINTENANCE_PATH)
        last = float(previous.get("lastRun", 0) or 0) if isinstance(previous, dict) else 0
        if not force and now - last < MAINTENANCE_INTERVAL_SECONDS:
            return previous if isinstance(previous, dict) else {"lastRun": last}
        entries = state.get("entries", {})
        expired = [
            key for key, entry in entries.items()
            if not isinstance(entry, dict) or now - float(entry.get("at", 0) or 0) > CACHE_TTL_SECONDS
        ]
        for key in expired:
            entries.pop(key, None)
        state["entries"] = entries
        state["maintenanceAt"] = now
        _save_cache(state)
        with _policy_lock:
            policy = _load_policy()
            adaptive = _recalibrate(policy, now)
            _compact_adaptive_state(policy)
            _save_policy(policy)
        record = {
            "version": 1,
            "lastRun": now,
            "nextRun": now + MAINTENANCE_INTERVAL_SECONDS,
            "expiredCacheEntries": len(expired),
            "policyVersion": policy.get("version", 1),
            "adaptiveUpdatedAt": adaptive.get("updatedAt", now),
            "adaptiveEventsConsidered": adaptive.get("eventsConsidered", 0),
        }
        temporary = CACHE_MAINTENANCE_PATH + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, CACHE_MAINTENANCE_PATH)
        return record


def _load_json(path):
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None
    except (OSError, ValueError, TypeError):
        return None


def maintenance_snapshot():
    return _load_json(CACHE_MAINTENANCE_PATH) or {"version": 1, "lastRun": 0, "nextRun": 0}


def _available(model, catalog):
    return model in catalog


def _healthy(model, policy, now=None):
    now = time.time() if now is None else now
    stats = policy["models"].setdefault(model, {"success": 0, "failure": 0, "quarantinedUntil": 0})
    return float(stats.get("quarantinedUntil", 0) or 0) <= now


def _failure_rate(model, policy):
    stats = policy["models"].setdefault(model, {"success": 0, "failure": 0, "quarantinedUntil": 0})
    total = int(stats.get("success", 0)) + int(stats.get("failure", 0))
    return (int(stats.get("failure", 0)) / total) if total else 0.0


def _pick_cheap(catalog, policy):
    return _pick_cheap_for_class(catalog, policy, None)


def _pick_cheap_for_class(catalog, policy, route_class=None):
    """Choose the least expensive healthy route that has earned promotion.

    Luna is the safe baseline. DeepSeek is promoted for a route class only
    after the persisted ledger has at least eight attempts and a 90% success
    rate for that class. A quarantined route is never selected, even when its
    recommendation is still present in the durable policy.
    """
    adaptive = policy.get("adaptive") or {}
    recommendations = adaptive.get("recommendations") or {}
    recommendation = recommendations.get(route_class or "default")
    recommended_model = recommendation.get("model") if isinstance(recommendation, dict) else None
    if recommended_model and _eligible(recommended_model, catalog, policy, route_class):
        return recommended_model
    for model in _candidate_order(route_class):
        if model == DEEPSEEK and not (
            isinstance(recommendation, dict) and recommendation.get("promoted") is True
        ):
            continue
        if _eligible(model, catalog, policy, route_class):
            return model
    return next(iter(catalog), LUNA)


def _candidate_order(route_class=None):
    """Return a capability-aware cheap-first order for a route class."""
    if route_class in ("consequential",):
        return (ASTRA, SOL, LUNA, DEEPSEEK)
    if route_class in ("hard_engineering", "recovery", "review"):
        return (SOL, ASTRA, LUNA, DEEPSEEK)
    return MODEL_ORDER


def _eligible(model, catalog, policy, route_class=None):
    """Keep learned failures from silently becoming the new default.

    Promotion is earned by the rolling adaptive ledger, while this short
    circuit protects against a provider degrading after the last hourly
    recalibration. Capability-sensitive classes never use an external
    route merely because it is cheap.
    """
    if not _available(model, catalog) or not _healthy(model, policy):
        return False
    if route_class in ("consequential", "hard_engineering", "recovery", "review"):
        return model in (ASTRA, SOL, LUNA)
    return _failure_rate(model, policy) <= MODEL_FAILURE_LIMIT


def classify(task, step, catalog):
    """Return a high-confidence local decision or None for Jev."""
    policy = _load_policy()
    if not policy.get("enabled", True):
        return None
    step_type = step.get("step_type")
    errored = bool(step.get("errored"))
    digest_len = len(step.get("digest") or "")
    task = (task or "").strip()
    context_chars = int(step.get("context_chars", 0) or 0)
    item_count = int(step.get("context_items", 0) or 0)
    file_count = len(FILE_RE.findall(task))
    if step_type == "tool_step":
        if errored:
            return {
                "route_class": "recovery",
                "model": SOL if _available(SOL, catalog) else ASTRA,
                "effort": "high",
                "confidence": 0.96,
                "reason": "tool_error",
            }
        if not step.get("deepseek_replay"):
            # OpenCode's DeepSeek thinking adapter requires the previous
            # assistant message's reasoning_content. Codex Responses history
            # generally exposes an encrypted reasoning item instead, which
            # the adapter cannot replay. Keep those tool steps on Luna so the
            # provider cannot reject the turn before a fallback can run.
            return {
                "route_class": "mechanical",
                "model": LUNA if _available(LUNA, catalog) else SOL,
                "effort": "medium",
                "confidence": 0.98,
                "reason": "missing_deepseek_reasoning_replay",
            }
        model = _pick_cheap_for_class(catalog, policy, "mechanical")
        return {
            "route_class": "mechanical",
            "model": model,
            "effort": "low" if digest_len <= 6000 else "medium",
            "confidence": 0.94,
            "reason": "clean_tool_continuation",
        }
    if step_type == "user_turn":
        if CRITICAL_RE.search(task):
            return {
                "route_class": "consequential",
                "model": ASTRA if _available(ASTRA, catalog) else SOL,
                "effort": "high",
                "confidence": 0.91,
                "reason": "risk_keyword",
            }
        if REVIEW_RE.search(task):
            return {
                "route_class": "review",
                "model": SOL if _available(SOL, catalog) else LUNA,
                "effort": "high",
                "confidence": 0.9,
                "reason": "review_or_audit",
            }
        if HARD_RE.search(task) or file_count >= 4 or context_chars >= 450000 or item_count >= 300:
            return {
                "route_class": "hard_engineering",
                "model": SOL if _available(SOL, catalog) else LUNA,
                "effort": "high",
                "confidence": 0.9,
                "reason": "hard_engineering_or_context_pressure",
            }
        if MECHANICAL_RE.search(task) and len(task) <= 800:
            return {
                "route_class": "routine",
                "model": _pick_cheap_for_class(catalog, policy, "routine"),
                "effort": "medium",
                "confidence": 0.87,
                "reason": "mechanical_keyword",
            }
    return None


def enrich_decision(decision, step=None, signals=None, cacheable=False):
    """Attach bounded metadata used by the shadow learner."""
    if not isinstance(decision, dict):
        return decision
    enriched = dict(decision)
    step = step if isinstance(step, dict) else {}
    signals = signals if isinstance(signals, dict) else {}
    enriched["features"] = {
        "route_class": enriched.get("route_class", "other"),
        "model": enriched.get("model", "unknown"),
        "effort": enriched.get("effort", "medium"),
        "confidence": enriched.get("confidence"),
        "reason": enriched.get("reason", "unknown"),
        "step_type": step.get("step_type", "other"),
        "context_chars": int(signals.get("context_chars", 0) or 0),
        "context_items": int(signals.get("context_items", 0) or 0),
        "errored": bool(step.get("errored")),
        "deepseek_replay": bool(step.get("deepseek_replay")),
        "phase_cacheable": bool(cacheable),
    }
    enriched["logisticProbability"] = logistic_predict(enriched["features"])
    return enriched


def record_outcome(decision, status):
    """Update route health after a completed relay and append a bounded event."""
    if not decision:
        return
    model = decision.get("model")
    if not model:
        return
    features = decision.get("features") if isinstance(decision, dict) else None
    if isinstance(features, dict):
        logistic_update(features, status)
    with _policy_lock:
        policy = _load_policy()
        stats = policy["models"].setdefault(model, {"success": 0, "failure": 0, "quarantinedUntil": 0})
        ok = int(status) == 200
        stats["success" if ok else "failure"] = int(stats.get("success", 0)) + 1
        attempts = int(stats["success"]) + int(stats["failure"])
        if (
            not ok
            and attempts >= int(policy.get("minAttemptsForQuarantine", 6))
            and _failure_rate(model, policy) >= float(policy.get("failureRateForQuarantine", 0.35))
        ):
            stats["quarantinedUntil"] = int(time.time() + int(policy.get("quarantineSeconds", 900)))
        _save_policy(policy)
        try:
            os.makedirs(STATE, mode=0o700, exist_ok=True)
            with open(EVENTS_PATH, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "routeClass": decision.get("route_class"),
                    "model": model,
                    "status": int(status),
                    "confidence": decision.get("confidence"),
                    "reason": decision.get("reason"),
                    "failureRate": round(_failure_rate(model, policy), 4),
                }) + "\n")
        except OSError:
            pass


def _event_timestamp(value):
    if not isinstance(value, str):
        return 0
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S").timestamp()
    except (TypeError, ValueError, OverflowError, OSError):
        return 0


def _event_stats(events):
    model_stats = {}
    class_stats = {}

    def add(bucket, model, ok):
        stats = bucket.setdefault(model, {"attempts": 0, "success": 0, "failure": 0})
        stats["attempts"] += 1
        stats["success" if ok else "failure"] += 1

    for event in events:
        if not isinstance(event, dict):
            continue
        model = event.get("model")
        route_class = event.get("routeClass") or "other"
        status = event.get("status")
        if not isinstance(model, str) or not model or not isinstance(status, int):
            continue
        ok = status == 200
        add(model_stats, model, ok)
        class_bucket = class_stats.setdefault(route_class, {})
        add(class_bucket, model, ok)

    def finish(bucket):
        result = {}
        for key, stats in bucket.items():
            attempts = int(stats["attempts"])
            success = int(stats["success"])
            failure = int(stats["failure"])
            result[key] = {
                "attempts": attempts,
                "success": success,
                "failure": failure,
                "successRate": round(success / attempts, 4) if attempts else 0.0,
            }
        return result

    return finish(model_stats), {
        route_class: {"models": finish(models)}
        for route_class, models in class_stats.items()
    }


def _recalibrate(policy, now=None):
    """Rebuild adaptive recommendations from the durable event ledger."""
    now = time.time() if now is None else now
    cutoff = now - ADAPTIVE_WINDOW_SECONDS
    events = []
    last_event_at = 0
    try:
        with open(EVENTS_PATH, encoding="utf-8") as handle:
            for line in handle:
                if len(events) >= ADAPTIVE_MAX_EVENTS:
                    events.pop(0)
                try:
                    event = json.loads(line)
                except (TypeError, ValueError):
                    continue
                if not isinstance(event, dict):
                    continue
                event_at = _event_timestamp(event.get("at"))
                if event_at and event_at < cutoff:
                    continue
                events.append(event)
                last_event_at = max(last_event_at, event_at)
    except OSError:
        pass

    model_stats, class_stats = _event_stats(events)
    recommendations = {}
    min_attempts = ADAPTIVE_MIN_ATTEMPTS
    min_rate = ADAPTIVE_MIN_SUCCESS_RATE
    # DeepSeek is the only route promoted below Luna. If it has not earned
    # promotion for a class, the caller remains on the Luna baseline.
    for route_class, details in class_stats.items():
        deepseek = (details.get("models") or {}).get(DEEPSEEK) or {}
        if int(deepseek.get("attempts", 0)) >= min_attempts and float(deepseek.get("successRate", 0)) >= min_rate:
            recommendations[route_class] = {
                "model": DEEPSEEK,
                "promoted": True,
                "attempts": int(deepseek["attempts"]),
                "successRate": float(deepseek["successRate"]),
                "reason": "earned_after_minimum_attempts_and_success_rate",
            }
        else:
            recommendations[route_class] = {
                "model": LUNA,
                "promoted": False,
                "attempts": int(deepseek.get("attempts", 0)),
                "successRate": float(deepseek.get("successRate", 0.0)),
                "reason": "waiting_for_minimum_attempts_and_success_rate",
            }

    policy["adaptive"] = {
        "updatedAt": now,
        "windowSeconds": ADAPTIVE_WINDOW_SECONDS,
        "minAttempts": min_attempts,
        "minSuccessRate": min_rate,
        "eventsConsidered": len(events),
        "lastEventAt": last_event_at,
        "models": model_stats,
        "classes": class_stats,
        "recommendations": recommendations,
    }
    return policy["adaptive"]


def _compact_adaptive_state(policy):
    """Keep the persisted policy bounded when old state comes from a prior build."""
    adaptive = policy.get("adaptive")
    if not isinstance(adaptive, dict):
        return
    for key in ("models", "classes", "recommendations"):
        value = adaptive.get(key)
        if not isinstance(value, dict):
            adaptive[key] = {}
        elif len(value) > 64:
            adaptive[key] = dict(list(value.items())[-64:])
    for model, stats in list(policy.get("models", {}).items()):
        if not isinstance(stats, dict):
            policy["models"].pop(model, None)
            continue
        for key in ("success", "failure", "quarantinedUntil"):
            stats[key] = int(stats.get(key, 0) or 0)


def snapshot():
    """Small dashboard-safe snapshot; no prompts or tool output are stored."""
    policy = _load_policy()
    _compact_adaptive_state(policy)
    return {
        "version": 1,
        "enabled": bool(policy.get("enabled", True)),
        "models": {
            model: {
                "success": int(stats.get("success", 0)),
                "failure": int(stats.get("failure", 0)),
                "failureRate": round(_failure_rate(model, policy), 4),
                "quarantinedUntil": int(stats.get("quarantinedUntil", 0) or 0),
            }
            for model, stats in policy.get("models", {}).items()
        },
        "adaptive": policy.get("adaptive") or {},
    }


def decision_context(catalog):
    """Return a small, prompt-safe state packet for Jev's next decision.

    This is deliberately aggregate-only: it contains no task text, tool
    output, file paths, or session identifiers. Jev can use current route
    health and earned promotions without receiving the local event ledger.
    """
    policy = _load_policy()
    adaptive = policy.get("adaptive") or {}
    models = {}
    for model in sorted(catalog):
        stats = policy.get("models", {}).get(model, {})
        models[model] = {
            "healthy": _healthy(model, policy),
            "failureRate": round(_failure_rate(model, policy), 4),
            "attempts": int(stats.get("success", 0) or 0) + int(stats.get("failure", 0) or 0),
        }
    recommendations = {}
    for route_class, recommendation in (adaptive.get("recommendations") or {}).items():
        if isinstance(recommendation, dict):
            recommendations[route_class] = {
                "model": recommendation.get("model"),
                "promoted": bool(recommendation.get("promoted", False)),
                "attempts": int(recommendation.get("attempts", 0) or 0),
                "successRate": float(recommendation.get("successRate", 0.0) or 0.0),
            }
    return {
        "policyVersion": int(policy.get("version", 1) or 1),
        "adaptiveWindowSeconds": int(adaptive.get("windowSeconds", ADAPTIVE_WINDOW_SECONDS) or ADAPTIVE_WINDOW_SECONDS),
        "models": models,
        "recommendations": recommendations,
        "jevCache": cache_snapshot(),
        "logistic": logistic_snapshot(),
    }
