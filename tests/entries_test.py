'''Tests for the Machine Learning Helpers.'''

import json
import unittest
from typing import List

from beancount.core.data import Transaction
from beancount.parser import parser

from smart_importer.entries import (add_suggested_accounts_to_transaction,
                                    add_posting_to_transaction)


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

    def test_add_predicted_posting_to_transaction(self):
        transaction: Transaction
        transaction = add_posting_to_transaction(self.test_transaction, "Expenses:Food:Groceries")
        self.assertEqual(transaction.postings[1].account, "Expenses:Food:Groceries")

    def test_add_suggested_accounts_to_transaction(self):
        suggestions: List[str]
        suggestions = ["Expenses:Food:Groceries",
                       "Expenses:Food:Restaurant",
                       "Expenses:Household",
                       "Expenses:Gifts"]
        transaction: Transaction
        transaction = add_suggested_accounts_to_transaction(self.test_transaction, suggestions)
        self.assertEqual(transaction.meta['__suggested_accounts__'], json.dumps(suggestions))
