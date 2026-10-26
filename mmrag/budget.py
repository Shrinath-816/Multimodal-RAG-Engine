"""
mmrag/budget.py
==================

Per-ticket spend cap, enforced BEFORE the next paid call (Week 8 section 18:
a check that only fires after the over-budget call has already spent the
money it was meant to prevent spending).
"""


class BudgetExceededError(Exception):
    """Raised when the next call's estimated cost would push the ticket over its cap."""


class TicketBudget:
    """Tracks spend for ONE ticket against a hard USD cap."""

    def __init__(self, max_cost_usd: float) -> None:
        self._max = max_cost_usd
        self._spent = 0.0

    @property
    def spent(self) -> float:
        """USD committed so far for this ticket."""
        return self._spent

    @property
    def remaining(self) -> float:
        """USD still available (never negative)."""
        return max(0.0, self._max - self._spent)

    def check(self, estimated_next_cost: float) -> None:
        """
        Raise BudgetExceededError if spending `estimated_next_cost` more would
        exceed the cap. Call BEFORE making the paid call.
        """
        if self._spent + estimated_next_cost > self._max:
            raise BudgetExceededError(
                f"Ticket budget ${self._max:.4f} would be exceeded: "
                f"${self._spent:.4f} spent + ${estimated_next_cost:.4f} estimated."
            )

    def commit(self, actual_cost: float) -> None:
        """Record ACTUAL cost after a call completes."""
        self._spent += actual_cost
