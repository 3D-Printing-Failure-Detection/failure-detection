"""Nothing reaches the printer without passing the gate. Pausing can be automatic (it is safe and
reversible on the P1S); resuming, correcting or stopping after an alarm needs a person unless the
approver is set to auto (simulation and tests only)."""


class AutoApprover:
    """Accepts every recommendation. For simulation and tests, never for a real print."""
    def choose(self, recommendation, options, summary):
        return recommendation["action"]


class TerminalApprover:
    """Shows the situation and the recommendation; Enter accepts, or type an option's number."""
    def __init__(self, ask=input, show=print):
        self.ask, self.show = ask, show

    def choose(self, recommendation, options, summary):
        self.show(summary)
        ids = [o["id"] for o in options]
        menu = "  ".join(f"[{i + 1}] {o}" for i, o in enumerate(ids))
        while True:
            a = self.ask(f"Recommended: {recommendation['action'].upper()}. Enter = accept, or {menu}: ").strip().lower()
            if not a:
                return recommendation["action"]
            if a.isdigit() and 1 <= int(a) <= len(ids):
                return ids[int(a) - 1]
            if a in ids:
                return a


class ActionGate:
    def __init__(self, tools, approver, auto_pause=True, log=print):
        self.tools, self.approver, self.auto_pause, self.log = tools, approver, auto_pause, log

    def pause_for_alarm(self) -> bool:
        if not self.auto_pause:
            return False
        self.tools.pause()
        return True

    def decide_and_act(self, recommendation, options, summary, correction_layers=None) -> str:
        action = self.approver.choose(recommendation, options, summary)
        if action not in {o["id"] for o in options}:
            raise ValueError(f"approver chose an option that was not offered: {action}")
        if action == "stop":
            self.tools.stop()
        elif action == "resume":
            self.tools.resume()
        elif action in ("correct", "restart"):
            self.tools.correct(correction_layers)
        return action                                # "continue": printer was never paused
