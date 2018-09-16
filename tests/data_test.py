"""Tests for the `PredictPostings` decorator"""

import io
import os
import unittest

from beancount.ingest.importer import ImporterProtocol
from beancount.parser import parser, printer

from smart_importer import PredictPostings


class PredictPostingsTest(unittest.TestCase):
    '''
    Data driven Tests for the `PredictPostings` decorator.
    '''

    def test_simple(self):
        '''
        Test with a simple testset.
        '''
        self.run_testset('simple')

    def test_multiaccounts(self):
        '''
        Test where the only differing factor is the account.
        '''
        self.run_testset('multiaccounts')

    def run_testset(self, testset):
        training_data = self.load_test_data(testset, 'training')
        extracted_data = self.load_test_data(testset, 'extracted')

        @PredictPostings(suggest=True)
        class DummyImporter(ImporterProtocol):
            def extract(self, file, existing_entries=None):
                return extracted_data

        importer = DummyImporter()
        imported_transactions = importer.extract(
            "dummy-data", existing_entries=training_data)
        with io.StringIO() as buffer:
            printer.print_entries(imported_transactions, file=buffer)
            actual = buffer.getvalue()

        expected_file_name = self.generate_file_name(testset, 'expected')
        if os.path.isfile(expected_file_name):
            with open(expected_file_name, 'r') as expected_file:
                expected = expected_file.read()
                self.assertEqual(expected, actual)
        else:
            with open(expected_file_name, 'w') as expected_file:
                expected_file.write(actual)

    @staticmethod
    def generate_file_name(testset, kind):
        return os.path.join(
            os.path.dirname(__file__), 'data',
            testset + '-' + kind + '.beancount')

    def load_test_data(self, testset, kind):
        filename = self.generate_file_name(testset, kind)
        data, errors, _ = parser.parse_file(filename)
        assert not errors
        return data
