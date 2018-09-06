"""Tests for the `PredictPostings` decorator"""

import unittest
from typing import List

from beancount.core.data import Transaction
from beancount.ingest.importer import ImporterProtocol
from beancount.parser import parser

from smart_importer import PredictPostings


class Testdata:
    test_data: List[Transaction]
    test_data, errors, _ = parser.parse_string("""
                2017-01-06 * "Farmer Fresh" "Buying groceries"
                  Assets:US:BofA:Checking  -2.50 USD

                2017-01-07 * "Farmer Fresh" "Groceries"
                  Assets:US:BofA:Checking  -10.20 USD

                2017-01-10 * "Uncle Boons" "Eating out with Joe"
                  Assets:US:BofA:Checking  -38.36 USD

                2017-01-10 * "Uncle Boons" "Dinner with Martin"
                  Assets:US:BofA:Checking  -35.00 USD

                2017-01-10 * "Walmarts" "Groceries"
                  Assets:US:BofA:Checking  -53.70 USD

                2017-01-10 * "Gimme Coffee" "Coffee"
                  Assets:US:BofA:Checking  -5.00 USD
                """)
    assert not errors

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
                """)
    assert not errors

    known_account = "Assets:US:BofA:Checking"

    correct_predictions = [
        'Expenses:Food:Groceries',
        'Expenses:Food:Groceries',
        'Expenses:Food:Restaurant',
        'Expenses:Food:Restaurant',
        'Expenses:Food:Groceries',
        'Expenses:Food:Coffee'
    ]


class BasicTestImporter(ImporterProtocol):
    def extract(self, file, existing_entries=None):
        return Testdata.test_data

    def file_account(self, file):
        return Testdata.known_account


class PredictPostingsTest(unittest.TestCase):
    '''
    Tests for machine learning functionality of the `PredictPostings` decorator.
    '''

    def setUp(self):
        '''
        Sets up the `PredictPostingsTest` unit test
        '''

        # define and decorate an importer:
        @PredictPostings(
            training_data=Testdata.training_data,
            account="Assets:US:BofA:Checking",
            suggest_accounts=True,
        )
        class DecoratedTestImporter(BasicTestImporter):
            pass

        self.importerClass = DecoratedTestImporter
        self.importer = DecoratedTestImporter()

    def test_unchanged_narrations(self):
        '''
        Verifies that the decorator leaves the narration intact
        '''
        correct_narrations = [transaction.narration for transaction in Testdata.test_data]
        extracted_narrations = [transaction.narration for transaction in self.importer.extract("dummy-data")]
        self.assertEqual(extracted_narrations, correct_narrations)

    def test_unchanged_first_posting(self):
        '''
        Verifies that the decorator leaves the first posting intact
        '''
        correct_first_postings = [transaction.postings[0] for transaction in Testdata.test_data]
        extracted_first_postings = [transaction.postings[0] for transaction in self.importer.extract("dummy-data")]
        self.assertEqual(extracted_first_postings, correct_first_postings)

    def test_predicted_postings(self):
        '''
        Verifies that the decorator adds predicted postings.
        '''
        transactions = self.importer.extract("dummy-data")
        predicted_accounts = [entry.postings[-1].account for entry in transactions]
        self.assertEqual(predicted_accounts, Testdata.correct_predictions)
        # print("Entries with predicted postings:")
        # printer.print_entries(entries)

    def test_added_suggestions(self):
        '''
        Verifies that the decorator adds suggestions about accounts
        that are likely to be involved in the transaction.
        '''
        transactions = self.importer.extract("dummy-data")
        for transaction in transactions:
            suggestions = transaction.meta['__suggested_accounts__']
            self.assertTrue(len(suggestions),
                            msg=f"The list of suggested accounts should not be empty, "
                                f"but was found to be empty for transaction {transaction}.")


class PredictPostingsDecorationTest(unittest.TestCase):
    '''
    Tests for the different variants how the decoration can be applied.
    '''

    def test_class_decoration_with_arguments(self):
        '''
        Verifies that the decorator can be applied to importer classes,
        with arguments supplied to the decorator.
        '''

        @PredictPostings(
            training_data=Testdata.training_data,
            account=Testdata.known_account
        )
        class SmartTestImporter(BasicTestImporter):
            pass

        i = SmartTestImporter()
        self.assertIsInstance(i, SmartTestImporter,
                              'The decorated importer shall still be an instance of the undecorated class.')
        transactions = i.extract('file', existing_entries=Testdata.training_data)
        predicted_accounts = [entry.postings[-1].account for entry in transactions]
        self.assertEqual(predicted_accounts, Testdata.correct_predictions)

    def test_method_decoration_with_arguments(self):
        '''
        Verifies that the decorator can be applied to an importer's extract method,
        with arguments supplied to the decorator.
        '''
        testcase = self

        class SmartTestImporter(BasicTestImporter):
            @PredictPostings(
                training_data=Testdata.training_data,
                account=Testdata.known_account
            )
            def extract(self, file, existing_entries=None):
                testcase.assertIsInstance(self, SmartTestImporter)
                return super().extract(file, existing_entries=existing_entries)

        i = SmartTestImporter()
        transactions = i.extract('file', existing_entries=Testdata.training_data)
        predicted_accounts = [entry.postings[-1].account for entry in transactions]
        self.assertEqual(predicted_accounts, Testdata.correct_predictions)

    def test_class_decoration_with_empty_arguments(self):
        '''
        Verifies that the decorator can be applied to importer classes,
        without supplying any arguments to the decorator.
        '''

        @PredictPostings()
        class SmartTestImporter(BasicTestImporter):
            pass

        i = SmartTestImporter()
        self.assertIsInstance(i, SmartTestImporter,
                              'The decorated importer shall still be an instance of the undecorated class.')
        transactions = i.extract('file', existing_entries=Testdata.training_data)
        predicted_accounts = [transaction.postings[-1].account for transaction in transactions]
        self.assertEqual(predicted_accounts, Testdata.correct_predictions)

    def test_method_decoration_with_empty_arguments(self):
        '''
        Verifies that the decorator can be applied to an importer's extract method,
        without supplying any arguments to the decorator.
        '''
        testcase = self

        class SmartTestImporter(BasicTestImporter):
            @PredictPostings()
            def extract(self, file, existing_entries=None):
                testcase.assertIsInstance(self, SmartTestImporter)
                return super().extract(file, existing_entries=existing_entries)

        i = SmartTestImporter()
        transactions = i.extract('file', existing_entries=Testdata.training_data)
        predicted_accounts = [entry.postings[-1].account for entry in transactions]
        self.assertEqual(predicted_accounts, Testdata.correct_predictions)
