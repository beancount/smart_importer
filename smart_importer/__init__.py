"""Smart importer for Beancount and Fava."""

from __future__ import annotations

from typing import TYPE_CHECKING

from smart_importer.entries import update_postings
from smart_importer.predictor import EntryPredictor

if TYPE_CHECKING:
    from beancount.core.data import Transaction


class PredictPayees(EntryPredictor):
    """Predicts payees."""

    attribute = "payee"
    weights = {"narration": 0.8, "payee": 0.5, "date.day": 0.1}


class PredictPostings(EntryPredictor):
    """Predicts posting accounts."""

    weights = {"narration": 0.8, "payee": 0.5, "date.day": 0.1}

    @property
    def targets(self) -> list[str]:
        assert self.training_data is not None
        return [
            " ".join(sorted(posting.account for posting in txn.postings))
            for txn in self.training_data
        ]

    def apply_prediction(
        self, entry: Transaction, prediction: str
    ) -> Transaction:
        return update_postings(entry, prediction.split(" "))
