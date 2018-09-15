"""Smart importer for Beancount and Fava."""

__version__ = '0.0.1'
__copyright__ = "Copyright (C) 2018 Johannes Harms"
__license__ = "MIT"

from smart_importer.entries import add_posting_to_transaction
from smart_importer.entries import add_suggestions_to_entry
from smart_importer.entries import set_entry_attribute
from smart_importer.pipelines import TxnPostingAccount
from smart_importer.predictor import SmartImporterDecorator


class PredictPayees(SmartImporterDecorator):
    """Suggest and predict payees."""

    attribute = 'payee'
    weights = {'narration': 0.8, 'payee': 0.5, 'date.day': 0.1}

    def __init__(self, predict=True, overwrite=False, suggest=False):
        super().__init__(predict, suggest)
        self.overwrite = overwrite

    def apply_prediction(self, entry, prediction):
        return set_entry_attribute(
            entry, self.attribute, prediction, overwrite=self.overwrite)

    def apply_suggestion(self, entry, suggestions):
        return add_suggestions_to_entry(
            entry, suggestions, key='__suggested_payees__')


class PredictPostings(SmartImporterDecorator):
    """Predict one missing posting."""

    weights = {
        'narration': 0.8,
        'payee': 0.5,
        'first_posting_account': 0.8,
        'date.day': 0.1,
    }

    def prepare_training_data(self):
        """
        Convert the training data into a list of `TxnPostingAccount` objects.
        """
        self.training_data = [
            TxnPostingAccount(t, p, pRef.account) for t in self.training_data
            for pRef in t.postings for p in t.postings
            if p.account != pRef.account
        ]

    @property
    def targets(self):
        return [txn.posting.account for txn in self.training_data]

    def apply_prediction(self, entry, prediction):
        return add_posting_to_transaction(entry, prediction)

    def apply_suggestion(self, entry, suggestions):
        return add_suggestions_to_entry(
            entry, suggestions, key='__suggested_accounts__')
