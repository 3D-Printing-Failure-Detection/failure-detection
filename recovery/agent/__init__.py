"""Agentic monitoring and recovery: printer tools, a pluggable decision maker, an approval gate.

  printer (Bambu P1S or simulator) --observations--> tools --> decider (rules or an LLM)
                                   <--approved action-- gate (auto for pause, a person otherwise)

The LLM slot is `LLMDecider(complete=...)`: pass any function (system, user) -> reply text.
"""
from .deciders import LLMDecider, RuleDecider  # noqa: F401
from .gate import ActionGate, AutoApprover, TerminalApprover  # noqa: F401
from .loop import run_agent  # noqa: F401
from .tools import PrinterTools  # noqa: F401
