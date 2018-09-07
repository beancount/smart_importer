'''Tests for the Machine Learning Helpers.'''

import json
import os
import unittest
from typing import List

from beancount.core.data import Transaction
from beancount.parser import parser

from smart_importer import machinelearning_helpers as ml


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
        test_data = ml.load_training_data(
            training_data=os.path.join(os.path.dirname(__file__), 'sample_training.beancount')
        )
        self.assertEqual(1, len(list(test_data)))

    def test_load_training_data_use_existing(self):
        existing_entries = self.test_data
        actual = ml.load_training_data(
            training_data=None,
            existing_entries=existing_entries
        )
        self.assertEqual(existing_entries, actual)

    def test_get_payee(self):
        self.assertEqual(ml.GetPayee().transform(self.test_data),
                         ['Farmer Fresh', 'Starbucks', 'Farmer Fresh', 'Gimme Coffee'])

    def test_get_payee2(self):
        self.assertEqual(ml.GetNarration().transform(self.test_data),
                         ['Buying groceries', 'Coffee', 'Groceries', 'Coffee'])

    def test_get_posting_account_of_transactions(self):
        self.assertEqual(ml.GetPostingAccount().transform(self.test_data),
                         ['Assets:US:BofA:Checking', 'Expenses:Food:Coffee', 'Expenses:Food:Groceries',
                          'Expenses:Food:Coffee'])

    def test_get_posting_account_of_txnpostingsaccount(self):
        txn_postings = [ml.TxnPostingAccount(t, p, 'Foo') for t in self.test_data for p in t.postings]
        self.assertEqual(ml.GetPostingAccount().transform(txn_postings),
                         ['Assets:US:BofA:Checking', 'Assets:US:BofA:Checking', 'Expenses:Food:Coffee',
                          'Assets:US:BofA:Checking', 'Expenses:Food:Groceries', 'Assets:US:BofA:Checking',
                          'Expenses:Food:Coffee'])

    def test_get_day_of_month(self):
        self.assertEqual(ml.GetDayOfMonth().transform(self.test_data), [6, 7, 7, 8])
