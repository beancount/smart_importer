"""Decorator for a Beancount importer that suggests and predicts payees."""

import logging
import operator
from typing import List, Union

from beancount.core.data import Transaction

from smart_importer.decorator import SmartImporterDecorator
from smart_importer.entries import add_payee_to_transaction
from smart_importer.entries import add_suggested_payees_to_transaction

logger = logging.getLogger(__name__)


class PredictPayees(SmartImporterDecorator):
    """
    Applying this decorator to a beancount importer or its extract method
    will predict and suggest payees
    of imported transactions.

    Predictions are implemented using machine learning
    based on training data read from a beancount file.

    Example:

    @PredictPayees()
    class MyImporter(ImporterProtocol):
        def extract(file):
          # do the import, return list of entries
    """

    def __init__(self,
                 training_data: Union[List[Transaction], str] = None,
                 account: str = None,
                 predict_payees: bool = True,
                 overwrite_existing_payees=False,
                 suggest_payees: bool = False):
        super().__init__()
        self.account = account
        self.training_data = training_data

        self.predict = predict_payees
        self.overwrite = overwrite_existing_payees
        self.suggest = suggest_payees

        self.weights = {'narration': 0.8, 'payee': 0.5, 'date.day': 0.1}

    @property
    def targets(self):
        return [txn.payee or '' for txn in self.training_data]

    def process_transactions(
            self, transactions: List[Transaction]) -> List[Transaction]:
        """Processes all imported transactions.

        Predict payees and suggest payees.
        """

        if self.predict:
            predictions = self.pipeline.predict(transactions)
            transactions = [
                add_payee_to_transaction(txn, payee, overwrite=self.overwrite)
                for txn, payee in zip(transactions, predictions)
            ]
            logger.debug("Added predictions to transactions.")

        if self.suggest:
            # get values from the SVC decision function
            decision_values = self.pipeline.decision_function(transactions)

            # add a human-readable class label (i.e., payee's name) to each
            # value, and sort by value:
            suggestions = [[
                payee for _, payee in sorted(
                    list(zip(distance_values, self.pipeline.classes_)),
                    key=operator.itemgetter(0),
                    reverse=True)
            ] for distance_values in decision_values]

            # Add the suggestions to each transaction:
            transactions = [
                add_suggested_payees_to_transaction(txn, payee)
                for txn, payee in zip(transactions, suggestions)
            ]
            logger.debug("Added suggestions to transactions.")
        return transactions
