"""Token counts and cost of every language-model call in a run (decide, plan, write G-code).

Prices are list prices in US dollars per million tokens (input, output), as of 2025-2026; check
https://openai.com/api/pricing and edit PRICES if they change. A model not listed is still
counted, with its cost shown as unknown.
"""
import json
import threading
import time

PRICES = {
    "gpt-4.1": (2.00, 8.00), "gpt-4.1-mini": (0.40, 1.60), "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o": (2.50, 10.00), "gpt-4o-mini": (0.15, 0.60),
    "gpt-5": (1.25, 10.00), "gpt-5-mini": (0.25, 2.00), "gpt-5-nano": (0.05, 0.40),
    "gpt-5.4": (2.50, 15.00), "gpt-5.4-mini": (0.75, 4.50), "gpt-5.4-nano": (0.20, 1.25),
}


def price_of(model):
    """(input, output) $/1M tokens; dated names (gpt-4.1-mini-2025-04-14) use their base name."""
    for name in sorted(PRICES, key=len, reverse=True):
        if model == name or model.startswith(name + "-2"):
            return PRICES[name]
    return None


class Usage:
    """Thread-safe log of model calls: who asked (label), which model, tokens in and out."""

    def __init__(self):
        self.calls, self._lock = [], threading.Lock()

    def record(self, model, label, prompt_tokens, completion_tokens, latency_s=None):
        p = price_of(model)
        cost = None if p is None else (prompt_tokens * p[0] + completion_tokens * p[1]) / 1e6
        with self._lock:
            self.calls.append(dict(t=time.time(), label=label or "llm", model=model,
                                   prompt_tokens=int(prompt_tokens), completion_tokens=int(completion_tokens),
                                   cost_usd=None if cost is None else round(cost, 6), latency_s=latency_s))

    def totals(self):
        with self._lock:
            calls = list(self.calls)
        out = {}
        for c in calls:
            t = out.setdefault(c["label"], dict(calls=0, prompt_tokens=0, completion_tokens=0, cost_usd=0.0))
            t["calls"] += 1
            t["prompt_tokens"] += c["prompt_tokens"]
            t["completion_tokens"] += c["completion_tokens"]
            t["cost_usd"] = None if t["cost_usd"] is None or c["cost_usd"] is None else t["cost_usd"] + c["cost_usd"]
        all_ = dict(calls=sum(t["calls"] for t in out.values()),
                    prompt_tokens=sum(t["prompt_tokens"] for t in out.values()),
                    completion_tokens=sum(t["completion_tokens"] for t in out.values()),
                    cost_usd=None if any(t["cost_usd"] is None for t in out.values())
                    else sum(t["cost_usd"] for t in out.values()))
        for t in list(out.values()) + [all_]:
            if t["cost_usd"] is not None:
                t["cost_usd"] = round(t["cost_usd"], 4)
        return dict(by_step=out, total=all_)

    def line(self):
        """One line for the terminal, e.g. 'LLM: 12 calls, 31,200 tokens in / 18,400 out, $0.042'."""
        t = self.totals()["total"]
        if not t["calls"]:
            return "LLM: no calls"
        cost = "cost unknown (model not in usage.PRICES)" if t["cost_usd"] is None else f"${t['cost_usd']:.4f}"
        steps = ", ".join(f"{k} {v['calls']}" for k, v in self.totals()["by_step"].items())
        return (f"LLM: {t['calls']} calls ({steps}), {t['prompt_tokens']:,} tokens in / "
                f"{t['completion_tokens']:,} out, {cost}")

    def save(self, path):
        with open(path, "w") as f:
            json.dump(dict(self.totals(), calls=self.calls), f, indent=1)


def labelled(complete, label):
    """The same model, with its calls counted under `label` (if it reports usage)."""
    if not getattr(complete, "counts_usage", False):
        return complete

    def fn(system, user, label=label):           # can be re-labelled again (decide -> plan)
        return complete(system, user, label=label)
    fn.counts_usage = True
    return fn
