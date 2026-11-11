"""Who recommends what to do after an alarm. Every decider sees the same situation (a JSON-able
dict) and must pick one of the allowed options; it never talks to the printer itself.

To plug in a language model, pass a `complete(system, user) -> text` function, e.g. with Claude:

    import anthropic
    client = anthropic.Anthropic()
    def complete(system, user):
        msg = client.messages.create(model=MODEL, max_tokens=400, system=system,
                                     messages=[{"role": "user", "content": user}])
        return msg.content[0].text
    decider = LLMDecider(complete)

or with any OpenAI-compatible local server (Ollama, llama.cpp, vLLM): see `openai_complete`.
A reply that is not valid JSON or names an option that is not allowed falls back to the next
decider (by default the rules), and the log says so. Chain them to get Claude -> local -> rules:

    LLMDecider(claude, name="claude", fallback=LLMDecider(local, name="local"))
"""
import json
import re
import time
import urllib.request

SYSTEM = """You supervise a 3D printer that just raised an alarm during a print. The print is
paused (unless stated otherwise). You get the measurement, a per-feature function check of the
finished part IF we simply continue, possibly a correction (new remaining layers that start on the
measured shape and return to the original) with its own function check, the filament cost, and the
allowed actions.
Goal: finish parts that still do their job even if they look different; stop parts that will not
work, since every further layer wastes filament. A cosmetic deviation alone is never a reason to
stop. Prefer continuing over correcting when both keep every feature working (correcting replaces
the print job, which is riskier). Do not invent measurements; rely on the function checks.
Reply with JSON only: {"action": "<one allowed action id>", "reason": "<one or two sentences>"}"""


class RuleDecider:
    name = "rules"

    def decide(self, situation: dict, options: list) -> dict:
        ids = {o["id"] for o in options}
        go_on = "resume" if "resume" in ids else "continue"
        if situation["observation"]["label"] == "detach":
            return dict(action="stop", reason="the part has come off the bed: nothing to build on",
                        source=self.name)
        report = situation["function_if_continue"]
        if report["continue_ok"]:
            return dict(action=go_on, reason="every functional feature still passes if we continue; "
                        "the deviation is cosmetic", source=self.name)
        bad = "; ".join(f"{f['feature']}: {f['detail']}" for f in report["features"] if not f["ok"])
        if "correct" in ids:
            fade = situation["correction"]["fade_layers"]
            back = f"returns to the original shape over {fade} layers" if fade else "keeps the new shape"
            return dict(action="correct", reason=f"continuing would fail ({bad}); the correction starts on "
                        f"the measured shape, {back}, and every feature passes", source=self.name)
        return dict(action="stop", reason="the finished part would fail and no correction passes: " + bad,
                    source=self.name)


def parse_reply(text: str) -> dict:
    """The JSON object in a model reply. Reasoning models (Qwen3, DeepSeek-R1) may emit a
    <think>...</think> block first, which can itself contain braces: drop it."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in reply")
    return json.loads(m.group(0))


def openai_complete(endpoint="http://127.0.0.1:11434", model="qwen3:1.7b", timeout=120):
    """complete(system, user) for an OpenAI-compatible chat server (Ollama's default port is
    11434). Deterministic, JSON-only replies; Qwen3's thinking is switched off with /no_think."""
    url = endpoint.rstrip("/") + "/v1/chat/completions"

    def complete(system, user):
        if model.lower().startswith("qwen3"):
            user = user + "\n/no_think"
        body = dict(model=model, temperature=0, response_format=dict(type="json_object"),
                    messages=[dict(role="system", content=system), dict(role="user", content=user)])
        req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]
    return complete


class LLMDecider:
    """A language model recommends; the fallback (rules, or another model) stands in whenever
    its reply is unusable."""

    def __init__(self, complete, fallback=None, system=SYSTEM, name="llm"):
        self.complete, self.fallback, self.system, self.name = complete, fallback or RuleDecider(), system, name
        self.transcript = []

    def decide(self, situation: dict, options: list) -> dict:
        user = json.dumps(dict(situation=situation, allowed_actions=options), indent=1, default=str)
        rec = dict(decider=self.name, prompt=user)
        t0 = time.time()
        try:
            reply = self.complete(self.system, user)
            rec["reply"] = reply
            ans = parse_reply(reply)
            if ans.get("action") not in {o["id"] for o in options}:
                raise ValueError(f"reply did not choose an allowed action: {reply[:200]!r}")
            out = dict(action=ans["action"], reason=str(ans.get("reason", "")), source=self.name)
        except Exception as e:              # bad reply, no key, server down... -> next decider, visibly
            rec["error"] = f"{type(e).__name__}: {e}"[:300]
            out = self.fallback.decide(situation, options)
            out["reason"] += f" ({self.name} unavailable or invalid: {type(e).__name__})"
            out["source"] = f"{self.name}->{out['source']}"
        rec["latency_s"] = round(time.time() - t0, 2)
        rec["decision"] = out
        self.transcript.append(rec)
        return out

    @property
    def full_transcript(self):
        """This decider's records plus those of any LLM fallback behind it."""
        return self.transcript + getattr(self.fallback, "full_transcript", [])
