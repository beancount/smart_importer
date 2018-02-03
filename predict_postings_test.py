"""Tests for the `PredictPostings` decorator"""

import unittest
from typing import List, Union

from beancount.core.data import ALL_DIRECTIVES, Transaction
from beancount.ingest.cache import _FileMemo
from beancount.ingest.importer import ImporterProtocol
from beancount.parser import printer, parser

from importers.smart_importer.predict_postings import PredictPostings


class PredictPostingsTest(unittest.TestCase):
    '''
    Tests for the PredictPostings decorator.
    '''

    def setUp(self):
        self.training_data: List[Transaction]
        self.training_data, errors, _ = parser.parse_string("""
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

        self.test_data, errors, _ = parser.parse_string("""
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

        self.correct_predictions = ['Expenses:Food:Groceries',
                                    'Expenses:Food:Groceries',
                                    'Expenses:Food:Restaurant',
                                    'Expenses:Food:Restaurant',
                                    'Expenses:Food:Groceries',
                                    'Expenses:Food:Coffee']

        # to be able to reference ourselves later on:
        testcase = self

        # define a test importer and decorate its extract function:
        class DummyImporter(ImporterProtocol):
            @PredictPostings(training_data=self.training_data,
                             filter_training_data_by_account="Assets:US:BofA:Checking")
            def extract(self, file: _FileMemo) -> List[Union[ALL_DIRECTIVES]]:
                return testcase.test_data

        self.importer = DummyImporter()

    def test_dummy_importer(self):
        '''
        Verifies the dummy importer
        '''
        print("\n\nRunning Test Case: {id}".format(id=self.id().split('.')[-1]))
        method_without_decorator = self.importer.extract.__wrapped__
        entries = method_without_decorator(self.importer, 'dummy-data')
        self.assertEqual(entries[0].narration, "Buying groceries")
        # print("Entries without predicted postings:")
        # printer.print_entries(entries)

    def test_unchanged_narrations(self):
        '''
        Verifies that the decorator leaves the narration intact
        '''
        print("\n\nRunning Test Case: {id}".format(id=self.id().split('.')[-1]))
        correct_narrations = [transaction.narration for transaction in self.test_data]
        extracted_narrations = [transaction.narration for transaction in self.importer.extract("dummy-data")]
        self.assertEqual(extracted_narrations, correct_narrations)

    def test_unchanged_first_posting(self):
        '''
        Verifies that the decorator leaves the first posting intact
        '''
        print("\n\nRunning Test Case: {id}".format(id=self.id().split('.')[-1]))
        correct_first_postings = [transaction.postings[0] for transaction in self.test_data]
        extracted_first_postings = [transaction.postings[0] for transaction in self.importer.extract("dummy-data")]
        self.assertEqual(extracted_first_postings, correct_first_postings)

    def test_predicted_postings(self):
        '''
        Verifies that the decorator adds predicted postings.
        '''
        print("\n\nRunning Test Case: {id}".format(id=self.id().split('.')[-1]))
        transactions = self.importer.extract("dummy-data")
        predicted_accounts = [entry.postings[-1].account for entry in transactions]
        self.assertEqual(predicted_accounts, self.correct_predictions)
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
                            msg=f"The list of suggested accounts should not be empty, but was found to be empty for transaction {transaction}.")


if __name__ == '__main__':
    unittest.main(buffer=True)
