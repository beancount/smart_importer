"""Decorator for Beancount importers that predicts postings."""

import logging
from typing import List, Union

from beancount.core.data import Transaction

from smart_importer.decorator import SmartImporterDecorator
from smart_importer.entries import add_posting_to_transaction
from smart_importer.entries import add_suggested_accounts_to_transaction
from smart_importer.pipelines import TxnPostingAccount

logger = logging.getLogger(__name__)


class PredictPostings(SmartImporterDecorator):
    """
    Applying this decorator to a beancount importer or its extract method
    will predict and suggest account names
    for imported transactions.

    Example:

    @PredictPostings(
        training_data="trainingdata.beancount",
        account="The:Importers:Already:Known:Accountname"
    )
    class MyImporter(ImporterProtocol):
        def extract(file):
          # do the import, return list of entries
    """

    def __init__(
            self,
            training_data: Union[List[Transaction], str] = None,
            account: str = None,
            predict: bool = True,
            suggest: bool = False,
    ):
        super().__init__()

        self.account = account
        self.training_data = training_data
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

    def process_transactions(
            self, transactions: List[Transaction]) -> List[Transaction]:
        """Process all imported transactions.

        * Predicts missing second postings.
        * Suggests accounts that are likely also involved in the transaction

        :param transactions: List of Beancount transactions
        :return: List of beancount transactions
        """
        if self.predict:
            logger.debug("Generate predictions for missing second postings.")
            predicted_accounts: List[str]
            predicted_accounts = self.pipeline.predict(transactions)
            transactions = [
                add_posting_to_transaction(txn, account)
                for txn, account in zip(transactions, predicted_accounts)
            ]
            logger.debug("Added predicted accounts.")
        if self.suggest:
            # get values from the SVC decision function
            logger.debug("Generate suggestions about related accounts.")
            decision_values = self.pipeline.decision_function(transactions)

            # add a human-readable class label (i.e., account name) to each
            # value, and sort by value:
            suggestions = [[
                account for _, account in sorted(
                    list(zip(distance_values, self.pipeline.classes_)),
                    key=lambda x: x[0],
                    reverse=True)
            ] for distance_values in decision_values]

            # add the suggested accounts to each transaction:
            transactions = [
                add_suggested_accounts_to_transaction(txn, suggestions)
                for txn, suggestions in zip(transactions, suggestions)
            ]
            logger.debug("Added suggested accounts.")
        return transactions
