'''Tests for the Machine Learning Helpers.'''

import json
import logging
import os
import unittest
from typing import List

from beancount.core.data import Transaction
from beancount.parser import parser

from smart_importer import machinelearning_helpers as ml

logger = logging.getLogger(__name__)


class MachinelearningTest(unittest.TestCase):
    '''
    Tests for machinelearning_helpers.py
    '''

    def setUp(self):
        '''
        Initialializes an importer where the PredictPostings decorator
        is applied to the extract function.
        '''
        self.test_data, errors, __ = parser.parse_string("""
                2016-01-06 * "Farmer Fresh" "Buying groceries"
                  Assets:US:BofA:Checking  -10.00 USD

                2016-01-07 * "Starbucks" "Coffee"
                  Assets:US:BofA:Checking  -4.00 USD
                  Expenses:Food:Coffee

                2016-01-07 * "Farmer Fresh" "Groceries"
                  Assets:US:BofA:Checking  -10.20 USD
                  Expenses:Food:Groceries

                2016-01-08 * "Gimme Coffee" "Coffee"
                  Assets:US:BofA:Checking  -3.50 USD
                  Expenses:Food:Coffee
                """)
        assert not errors
        self.test_transaction: Transaction
        self.test_transaction = self.test_data[0]

    def test_load_training_data(self):
        logger.info("Running Test Case: {id}".format(id=self.id().split('.')[-1]))
        test_data = ml.load_training_data(
            training_data=os.path.join(os.path.dirname(__file__), 'sample_training.beancount')
        )
        self.assertEqual(1, len(list(test_data)))

    def test_load_training_data_use_existing(self):
        logger.info("Running Test Case: {id}".format(id=self.id().split('.')[-1]))
        existing_entries = self.test_data
        actual = ml.load_training_data(
            training_data=None,
            existing_entries=existing_entries
        )
        self.assertEqual(existing_entries, actual)

    def test_transaction_involves_account(self):
        logger.info("Running Test Case: {id}".format(id=self.id().split('.')[-1]))
        self.assertTrue(ml.transaction_involves_account(self.test_transaction, None))
        self.assertTrue(ml.transaction_involves_account(self.test_transaction, 'Assets:US:BofA:Checking'))
        self.assertFalse(ml.transaction_involves_account(self.test_transaction, 'Some:Unknown:Account'))

    def test_add_predicted_posting_to_transaction(self):
        logger.info("Running Test Case: {id}".format(id=self.id().split('.')[-1]))
        transaction: Transaction
        transaction = ml.add_posting_to_transaction(self.test_transaction, "Expenses:Food:Groceries")
        self.assertEqual(transaction.postings[1].account, "Expenses:Food:Groceries")

    def test_add_suggested_accounts_to_transaction(self):
        logger.info("Running Test Case: {id}".format(id=self.id().split('.')[-1]))
        suggestions: List[str]
        suggestions = ["Expenses:Food:Groceries",
                       "Expenses:Food:Restaurant",
                       "Expenses:Household",
                       "Expenses:Gifts"]
        transaction: Transaction
        transaction = ml.add_suggested_accounts_to_transaction(self.test_transaction, suggestions)
        self.assertEqual(transaction.meta['__suggested_accounts__'], json.dumps(suggestions))

    def test_get_payee(self):
        logger.info("Running Test Case: {id}".format(id=self.id().split('.')[-1]))
        self.assertEqual(ml.GetPayee().transform(self.test_data),
                         ['Farmer Fresh', 'Starbucks', 'Farmer Fresh', 'Gimme Coffee'])

    def test_get_payee2(self):
        logger.info("Running Test Case: {id}".format(id=self.id().split('.')[-1]))
        self.assertEqual(ml.GetNarration().transform(self.test_data),
                         ['Buying groceries', 'Coffee', 'Groceries', 'Coffee'])

    def test_get_posting_account_of_transactions(self):
        logger.info("Running Test Case: {id}".format(id=self.id().split('.')[-1]))
        self.assertEqual(ml.GetPostingAccount().transform(self.test_data),
                         ['Assets:US:BofA:Checking', 'Expenses:Food:Coffee', 'Expenses:Food:Groceries',
                          'Expenses:Food:Coffee'])

    def test_get_posting_account_of_txnpostingsaccount(self):
        logger.info("Running Test Case: {id}".format(id=self.id().split('.')[-1]))
        txn_postings = [ml.TxnPostingAccount(t, p, 'Foo') for t in self.test_data for p in t.postings]
        self.assertEqual(ml.GetPostingAccount().transform(txn_postings),
                         ['Assets:US:BofA:Checking', 'Assets:US:BofA:Checking', 'Expenses:Food:Coffee',
                          'Assets:US:BofA:Checking', 'Expenses:Food:Groceries', 'Assets:US:BofA:Checking',
                          'Expenses:Food:Coffee'])

    def test_get_day_of_month(self):
        logger.info("Running Test Case: {id}".format(id=self.id().split('.')[-1]))
        self.assertEqual(ml.GetDayOfMonth().transform(self.test_data), [6, 7, 7, 8])


if __name__ == '__main__':
    # configure the log level
    LOG_LEVEL = logging.DEBUG
    logging.basicConfig(level=LOG_LEVEL)

    # colorize the log output if the coloredlogs package is available
    try:
        import coloredlogs
    except ImportError as e:
        coloredlogs = None
    if coloredlogs:
        coloredlogs.install(level=LOG_LEVEL)

    # show test case execution output iff logging level is DEBUG or finer:
    show_output = LOG_LEVEL <= logging.DEBUG
    unittest.main(buffer=not show_output)
