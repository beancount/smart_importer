"""Test cases for the `PredictPostings` decorator"""

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
        entries = method_without_decorator(self.importer, 'this-is-never-read-by-the-dummy-importer.csv')
        self.assertEqual(entries[0].narration, "Buying groceries")
        # print("Entries without predicted postings:")
        # printer.print_entries(entries)

    def test_unchanged_narration(self):
        '''
        Verifies that the decorator leaves the narration intact
        '''
        print("\n\nRunning Test Case: {id}".format(id=self.id().split('.')[-1]))
        

    def test_predicted_postings(self):
        '''
        Tests the importer with predicted postings.
        '''
        print("\n\nRunning Test Case: {id}".format(id=self.id().split('.')[-1]))
        entries = self.importer.extract("this-is-never-read-by-the-dummy-importer.csv")

        print("Entries with predicted postings:")
        printer.print_entries(entries)

        self.assertEqual(entries[0].postings[1].account, "Expenses:Food:Groceries")


if __name__ == '__main__':
    unittest.main(buffer=True)
