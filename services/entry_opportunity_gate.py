"""Exchange and symbol safety boundary for production entries.

Return quality and sizing are adjudicated elsewhere. This policy intentionally
contains no score, evidence, probe, cooldown, exposure, or market thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ai_brain.base_model import DecisionOutput


@dataclass(frozen=True, slots=True)
class EntryOpportunityGatePolicy:
    """Block only symbols that cannot safely be submitted to the exchange."""

    suspicious_symbol_policy: Any | None = None

    def safety_reason(self, decision: DecisionOutput) -> str | None:
        if self.suspicious_symbol_policy is not None:
            reason = self.suspicious_symbol_policy.reason(decision.symbol)
            if reason:
                return str(reason)
        return None
