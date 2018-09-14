"""Decorator for Beancount importers that predicts postings."""

import logging

from smart_importer.entries import add_posting_to_transaction
from smart_importer.entries import add_suggested_accounts_to_transaction
from smart_importer.pipelines import TxnPostingAccount
from smart_importer.predictor import SmartImporterDecorator

logger = logging.getLogger(__name__)


class PredictPostings(SmartImporterDecorator):
    """Predict one missing posting."""

    def __init__(
            self,
            account: str = None,
            predict: bool = True,
            suggest: bool = False,
    ):
        super().__init__(predict, suggest)

        self.account = account
        self.predict = predict
        self.suggest = suggest

        self.weights = {
            'narration': 0.8,
            'first_posting_account': 0.8,
            'date.day': 0.1,
        }

    def prepare_training_data(self):
        """Prepare the training data.

        Convert the training data into a list of `TxnPostingAccount` objects.
        This list contains tuples of:

        * one transaction (for each transaction within the training data),
        * one posting (for each posting in every transaction), primarily used
          for that posting's account name,
        * and one other account name (for each other account in the same
          transaction).
        """
        self.training_data = [
            TxnPostingAccount(t, p, pRef.account) for t in self.training_data
            for pRef in t.postings for p in t.postings
            if p.account != pRef.account
        ]
        distinct_payees = set(
            map(lambda trx: trx.txn.payee, self.training_data or []))
        if len(distinct_payees) > 1:
            self.weights['payee'] = 0.5
        elif 'payee' in self.weights:
            del self.weights['payee']

    @property
    def targets(self):
        return [txn.posting.account for txn in self.training_data]

    def apply_prediction(self, entry, prediction):
        return add_posting_to_transaction(entry, prediction)

    def apply_suggestion(self, entry, suggestions):
        return add_suggested_accounts_to_transaction(entry, suggestions)
