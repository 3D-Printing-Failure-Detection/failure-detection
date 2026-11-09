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

or with any OpenAI-compatible local server (llama.cpp, Ollama, vLLM) the same way. A reply that
is not valid JSON or names an option that is not allowed falls back to the rules, and says so.
"""
import json
import re

SYSTEM = """You supervise a 3D printer that just raised an alarm during a print. The print is
paused (unless stated otherwise). You get the camera measurement, a per-feature function check of
the finished part IF we simply continue, the filament cost, and the allowed actions.
Goal: finish parts that still do their job even if they look different; stop parts that will not
work, since every further layer wastes filament. A cosmetic deviation alone is never a reason to
stop. Do not invent measurements; rely on the function check.
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
        bad = [f"{f['feature']}: {f['detail']}" for f in report["features"] if not f["ok"]]
        return dict(action="stop", reason="the finished part would fail: " + "; ".join(bad),
                    source=self.name)


class LLMDecider:
    """A language model recommends; the rules stand in whenever its reply is unusable."""
    name = "llm"

    def __init__(self, complete, fallback=None, system=SYSTEM):
        self.complete, self.fallback, self.system = complete, fallback or RuleDecider(), system
        self.transcript = []

    def decide(self, situation: dict, options: list) -> dict:
        user = json.dumps(dict(situation=situation, allowed_actions=options), indent=1, default=str)
        rec = dict(prompt=user)
        try:
            reply = self.complete(self.system, user)
            rec["reply"] = reply
            m = re.search(r"\{.*\}", reply, re.S)
            ans = json.loads(m.group(0)) if m else None
            if not ans or ans.get("action") not in {o["id"] for o in options}:
                raise ValueError(f"reply did not choose an allowed action: {reply[:200]!r}")
            out = dict(action=ans["action"], reason=str(ans.get("reason", "")), source=self.name)
        except Exception as e:                      # bad reply, no key, network... -> rules, visibly
            out = self.fallback.decide(situation, options)
            out["reason"] += f" (model unavailable or invalid: {type(e).__name__})"
            out["source"] = f"{self.name}->rules"
        rec["decision"] = out
        self.transcript.append(rec)
        return out
