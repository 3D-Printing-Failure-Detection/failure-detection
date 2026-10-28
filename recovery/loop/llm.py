"""Optional LLM planner. It sees a JSON summary of the rule-verified candidates and picks
one by name. Anything it returns that is not a spec-passing candidate is ignored and the
rule policy's choice is used, so the LLM can only choose *between safe options*."""
import json
import os

from .decide import Decision, RulePolicy

MODEL = "claude-sonnet-5-5"

SYSTEM = """You supervise a 3D printer that has just detected a failure mid-print.
You get candidate recovery plans, each already checked against the part's function spec.
Choose the best action for the user: prefer finishing a functional part, then prefer plans
that stay closest to the original design (higher fidelity), then saving filament/time.
Reply with JSON only: {"choice": "<candidate name or scrap>", "why": "<one sentence>"}."""


class LLMPolicy(RulePolicy):
    def __init__(self, *a, client=None, **kw):
        super().__init__(*a, **kw)
        self.client = client

    def _client(self):
        if self.client is None:
            import anthropic
            self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        return self.client

    def pick(self, cands) -> Decision:
        rule = super().pick(cands)
        try:
            summary = dict(failure_candidates=rule.considered, rule_choice=rule.plan.name if rule.plan else "scrap")
            msg = self._client().messages.create(
                model=MODEL, max_tokens=300, system=SYSTEM,
                messages=[{"role": "user", "content": json.dumps(summary)}])
            ans = json.loads(msg.content[0].text)
            choice = ans["choice"]
        except Exception as e:                      # no key / network / bad JSON -> rules
            rule.reason += f" (LLM unavailable: {type(e).__name__})"
            return rule
        if choice == "scrap":
            return Decision("scrap", "LLM: " + ans.get("why", ""), None, rule.verdict, rule.considered)
        for c in cands:
            if c["name"] == choice and c["verdict"].ok:
                act = "continue" if choice == "continue" else "adapt"
                return Decision(act, "LLM: " + ans.get("why", ""), c["plan"], c["verdict"], rule.considered)
        return rule
