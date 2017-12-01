"""Test cases for the `PredictPostings` decorator"""

import unittest
from typing import List, Union

from beancount.core.data import ALL_DIRECTIVES
from beancount.ingest.cache import _FileMemo
from beancount.ingest.importer import ImporterProtocol
from beancount.parser import printer, parser

from importers.smart_importer.predict_postings import PredictPostings


class PredictPostingsTest(unittest.TestCase):
    '''
    Test case for the PredictPostings decorator.
    '''

    def setUp(self):
        '''
        Initialializes an importer where the PredictPostings decorator
        is applied to the extract function.
        '''

        training_data, errors, _ = parser.parse_string("""
            2016-01-06 * "Farmer Fresh" "Buying groceries"
              Assets:US:BofA:Checking  -2.50 USD
              Expenses:Food:Groceries
            
            2016-01-07 * "Farmer Fresh" "Groceries"
              Assets:US:BofA:Checking  -10.20 USD
              Expenses:Food:Groceries
            
            2016-01-10 * "Uncle Boons" "Eating out with Joe"
              Assets:US:BofA:Checking  -38.36 USD
              Expenses:Food:Restaurant
            
            2016-01-10 * "Uncle Boons" "Dinner with Mary"
              Assets:US:BofA:Checking  -35.00 USD
              Expenses:Food:Restaurant
            
            2016-01-10 * "Walmarts" "Groceries"
              Assets:US:BofA:Checking  -53.70 USD
              Expenses:Food:Groceries
            """)
        assert not errors

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
            """)
        assert not errors

        class MyDemoImporter(ImporterProtocol):
            @PredictPostings(training_data, "Assets:US:BofA:Checking")
            def extract(self, file: _FileMemo) -> List[Union[ALL_DIRECTIVES]]:
                return test_data

        self.importer = MyDemoImporter()

    def test_extract_without_predictions(self):
        '''
        Tests the importer without predicted postings,
        i.e., as if the decorator had not been applied.
        '''
        print("\n\nRunning Test Case: {id}".format(id=self.id().split('.')[-1]))
        method_without_decorator = self.importer.extract.__wrapped__
        entries = method_without_decorator(self.importer, 'bank.csv')

        print("Entries without predicted postings:")
        printer.print_entries(entries)

        first_entry = entries[0]
        # self.assertEqual(first_entry, "Statement 1 imported from bank.csv")

    def test_predicted_postings(self):
        '''
        Tests the importer with predicted postings.
        '''
        print("\n\nRunning Test Case: {id}".format(id=self.id().split('.')[-1]))
        entries = self.importer.extract("bank.csv")

        print("Entries with predicted postings:")
        printer.print_entries(entries)

        # first_entry = entries[0]
        # self.assertEqual(first_entry,
        #                  "Statement 1 imported from bank.csv, "
        #                  "with a posting predicted by a model trained on training_data.beancount")


if __name__ == '__main__':
    unittest.main(buffer=True)
