"""Smart importer for Beancount and Fava."""
from smart_importer.entries import add_suggestions_to_entry
from smart_importer.entries import update_postings
from smart_importer.hooks import apply_hooks  # noqa
from smart_importer.predictor import EntryPredictor


class PredictPayees(EntryPredictor):
    """Suggest and predict payees."""

    attribute = "payee"
    weights = {"narration": 0.8, "payee": 0.5, "date.day": 0.1}


class PredictPostings(EntryPredictor):
    """Suggest and predict posting accounts."""

    weights = {"narration": 0.8, "payee": 0.5, "date.day": 0.1}

    @property
    def targets(self):
        return [
            " ".join(posting.account for posting in txn.postings)
            for txn in self.training_data
        ]

    def apply_prediction(self, entry, prediction):
        return update_postings(entry, prediction.split(" "))

    def apply_suggestion(self, entry, suggestions):
        accounts = dict()
        for accounts_string in suggestions:
            for account in accounts_string.split(" "):
                accounts[account] = None

        return add_suggestions_to_entry(
            entry, list(accounts.keys()), key="__suggested_accounts__"
        )
