"""Tests for the `PredictPostings` decorator"""

import unittest
from typing import List

from beancount.core.data import Transaction
from beancount.ingest.importer import ImporterProtocol
from beancount.parser import parser

from smart_importer.entries import METADATA_KEY_SUGGESTED_PAYEES
from smart_importer.predict_payees import PredictPayees


class Testdata:
    test_data, errors, _ = parser.parse_string("""
        2017-01-06 * "Farmer Fresh" "Buying groceries"
          Assets:US:BofA:Checking  -2.50 USD

        2017-01-07 * "Groceries"
          Assets:US:BofA:Checking  -10.20 USD

        2017-01-10 * "" "Eating out with Joe"
          Assets:US:BofA:Checking  -38.36 USD

        2017-01-10 * "Dinner with Martin"
          Assets:US:BofA:Checking  -35.00 USD

        2017-01-10 * "Groceries"
          Assets:US:BofA:Checking  -53.70 USD

        2017-01-10 * "Gimme Coffee" "Coffee"
          Assets:US:BofA:Checking  -5.00 USD
        """)

    training_data: List[Transaction]
    training_data, errors, _ = parser.parse_string("""
        2016-01-06 * "Farmer Fresh" "Buying groceries"
          Assets:US:BofA:Checking  -2.50 USD
          Expenses:Food:Groceries

        2016-01-07 * "Starbucks" "Coffee"
          Assets:US:BofA:Checking  -4.00 USD
          Expenses:Food:Coffee

        2016-01-07 * "Farmer Fresh" "Groceries"
          Assets:US:BofA:Checking  -10.20 USD
          Expenses:Food:Groceries

        2016-01-07 * "Gimme Coffee" "Coffee"
          Assets:US:BofA:Checking  -3.50 USD
          Expenses:Food:Coffee

        2016-01-08 * "Uncle Boons" "Eating out with Joe"
          Assets:US:BofA:Checking  -38.36 USD
          Expenses:Food:Restaurant

        2016-01-10 * "Walmarts" "Groceries"
          Assets:US:BofA:Checking  -53.70 USD
          Expenses:Food:Groceries

        2016-01-10 * "Gimme Coffee" "Coffee"
          Assets:US:BofA:Checking  -6.19 USD
          Expenses:Food:Coffee

        2016-01-10 * "Uncle Boons" "Dinner with Mary"
          Assets:US:BofA:Checking  -35.00 USD
          Expenses:Food:Restaurant

        2016-01-11 * "Farmer Fresh" "Groceries"
          Assets:US:BofA:Checking  -30.50 USD
          Expenses:Food:Groceries
        """)

    known_account = "Assets:US:BofA:Checking"

    correct_predictions = [
        'Farmer Fresh', 'Farmer Fresh', 'Uncle Boons', 'Uncle Boons',
        'Farmer Fresh', 'Gimme Coffee'
    ]


class BasicTestImporter(ImporterProtocol):
    def extract(self, file, existing_entries=None):
        return Testdata.test_data

    def file_account(self, file):
        return Testdata.known_account


class PredictPayeesTest(unittest.TestCase):
    '''
    Tests for machine learning functionality of the `PredictPayees` decorator.
    '''

    def setUp(self):
        '''
        Sets up the `PredictPayeesTest` unit test
        '''

        # define and decorate an importer:
        @PredictPayees(
            training_data=Testdata.training_data,
            account="Assets:US:BofA:Checking",
            overwrite_existing_payees=False,
            suggest_payees=True,
        )
        class DecoratedTestImporter(BasicTestImporter):
            pass

        self.importerClass = DecoratedTestImporter
        self.importer = DecoratedTestImporter()

    def test_dummy_importer(self):
        '''
        Verifies the dummy importer
        '''
        undecorated_importer = super(self.importerClass, self.importer)
        entries = undecorated_importer.extract('dummy-data')
        self.assertEqual(entries[0].narration, "Buying groceries")

    def test_unchanged_narrations(self):
        '''
        Verifies that the decorator leaves the narration intact
        '''
        correct_narrations = [
            transaction.narration for transaction in Testdata.test_data
        ]
        extracted_narrations = [
            transaction.narration
            for transaction in self.importer.extract("dummy-data")
        ]
        self.assertEqual(extracted_narrations, correct_narrations)

    def test_unchanged_first_posting(self):
        '''
        Verifies that the decorator leaves the first posting intact
        '''
        correct_first_postings = [
            transaction.postings[0] for transaction in Testdata.test_data
        ]
        extracted_first_postings = [
            transaction.postings[0]
            for transaction in self.importer.extract("dummy-data")
        ]
        self.assertEqual(extracted_first_postings, correct_first_postings)

    def test_predicted_payees(self):
        '''
        Verifies that the decorator adds predicted postings.
        '''
        transactions = self.importer.extract("dummy-data")
        predicted_payees = [transaction.payee for transaction in transactions]
        self.assertEqual(predicted_payees, Testdata.correct_predictions)

    def test_added_suggestions(self):
        '''
        Verifies that the decorator adds suggestions about accounts
        that are likely to be involved in the transaction.
        '''
        transactions = self.importer.extract("dummy-data")
        for transaction in transactions:
            suggestions = transaction.meta[METADATA_KEY_SUGGESTED_PAYEES]
            self.assertTrue(
                len(suggestions),
                msg=f"The list of suggested accounts should not be empty, "
                f"but was found to be empty for transaction {transaction}.")
