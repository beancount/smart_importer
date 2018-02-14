"""Tests for the `PredictPostings` decorator"""

import logging
import unittest
from typing import List

from beancount.core.data import Transaction
from beancount.ingest.cache import _FileMemo
from beancount.ingest.importer import ImporterProtocol
from beancount.parser import parser

from smart_importer import machinelearning_helpers as ml
from smart_importer.predict_payees import PredictPayees

LOG_LEVEL = logging.DEBUG
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

# colorize the log output if the coloredlogs package is available
try:
    import coloredlogs
except ImportError as e:
    coloredlogs = None
if coloredlogs:
    coloredlogs.install(level=LOG_LEVEL)


class PredictPayeesTest(unittest.TestCase):
    '''
    Tests for the `PredictPayees` decorator.
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

            2016-01-11 * "Farmer Fresh" "Groceries"
              Assets:US:BofA:Checking  -30.50 USD
              Expenses:Food:Groceries
            """)
        assert not errors

        self.test_data, errors, _ = parser.parse_string("""
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
        assert not errors

        self.correct_predictions = ['Farmer Fresh',
                                    'Farmer Fresh',
                                    'Uncle Boons',
                                    'Uncle Boons',
                                    'Farmer Fresh',
                                    'Gimme Coffee']

        # to be able to reference ourselves later on:
        testcase = self

        # define a test importer and decorate its extract function:
        class DummyImporter(ImporterProtocol):
            @PredictPayees(training_data=self.training_data,
                           filter_training_data_by_account="Assets:US:BofA:Checking",
                           overwrite_existing_payees=False)
            def extract(self, file: _FileMemo) -> List[Transaction]:
                return testcase.test_data

        self.importer = DummyImporter()

    def test_dummy_importer(self):
        '''
        Verifies the dummy importer
        '''
        logger.info("Running Test Case: {id}".format(id=self.id().split('.')[-1]))
        method_without_decorator = self.importer.extract.__wrapped__
        entries = method_without_decorator(self.importer, 'dummy-data')
        self.assertEqual(entries[0].narration, "Buying groceries")
        # print("Entries without predicted postings:")
        # printer.print_entries(entries)

    def test_unchanged_narrations(self):
        '''
        Verifies that the decorator leaves the narration intact
        '''
        logger.info("Running Test Case: {id}".format(id=self.id().split('.')[-1]))
        correct_narrations = [transaction.narration for transaction in self.test_data]
        extracted_narrations = [transaction.narration for transaction in self.importer.extract("dummy-data")]
        self.assertEqual(extracted_narrations, correct_narrations)

    def test_unchanged_first_posting(self):
        '''
        Verifies that the decorator leaves the first posting intact
        '''
        logger.info("Running Test Case: {id}".format(id=self.id().split('.')[-1]))
        correct_first_postings = [transaction.postings[0] for transaction in self.test_data]
        extracted_first_postings = [transaction.postings[0] for transaction in self.importer.extract("dummy-data")]
        self.assertEqual(extracted_first_postings, correct_first_postings)

    def test_predicted_payees(self):
        '''
        Verifies that the decorator adds predicted postings.
        '''
        logger.info("Running Test Case: {id}".format(id=self.id().split('.')[-1]))
        transactions = self.importer.extract("dummy-data")
        predicted_payees = [entry.payee for entry in transactions]
        self.assertEqual(predicted_payees, self.correct_predictions)

    def test_added_suggestions(self):
        '''
        Verifies that the decorator adds suggestions about accounts
        that are likely to be involved in the transaction.
        '''
        logger.info("Running Test Case: {id}".format(id=self.id().split('.')[-1]))
        transactions = self.importer.extract("dummy-data")
        for transaction in transactions:
            suggestions = transaction.meta[ml.METADATA_KEY_SUGGESTED_PAYEES]
            self.assertTrue(len(suggestions),
                            msg=f"The list of suggested accounts should not be empty, "
                                f"but was found to be empty for transaction {transaction}.")


if __name__ == '__main__':
    # show test case execution output iff logging level is DEBUG or finer:
    show_output = LOG_LEVEL <= logging.DEBUG
    unittest.main(buffer=not show_output)
