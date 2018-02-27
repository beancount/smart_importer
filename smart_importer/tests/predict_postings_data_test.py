"""Tests for the `PredictPostings` decorator"""

import logging
import unittest
import os
import io
from typing import List, Union

from beancount.core.data import ALL_DIRECTIVES, Transaction
from beancount.ingest.cache import _FileMemo
from beancount.ingest.importer import ImporterProtocol
from beancount.parser import parser, printer

from smart_importer.predict_postings import PredictPostings

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


class PredictPostingsTest(unittest.TestCase):
    '''
    Data driven Tests for the `PredictPostings` decorator.
    '''

    def test_simple(self):
        '''
        Test with a simple testset.
        '''
        logger.info("Running Test Case: {id}".format(id=self.id().split('.')[-1]))
        self.run_testset('simple')

    def run_testset(self, testset):
        training_data = self.load_test_data(testset, 'training') 
        extracted_data = self.load_test_data(testset, 'extracted') 

        class DummyImporter(ImporterProtocol):
            @PredictPostings(suggest_accounts=True)
            def extract(self, file: _FileMemo, existing_entries: List[Union[ALL_DIRECTIVES]]) -> List[Union[ALL_DIRECTIVES]]:
                return extracted_data

        importer = DummyImporter()
        actualTrxs = importer.extract("dummy-data", existing_entries=training_data)
        with io.StringIO() as buffer:
            printer.print_entries(actualTrxs, file=buffer)
            actual = buffer.getvalue()

        expected_file_name = self.generate_file_name(testset, 'expected')
        if os.path.isfile(expected_file_name):
            with open(expected_file_name, 'r') as expected_file:
                expected = expected_file.read()
                self.assertEqual(expected, actual)
        else:
            with open(expected_file_name, 'w') as expected_file:
                expected_file.write(actual)


    def generate_file_name(self, testset, kind):
        return os.path.join(os.path.dirname(__file__), 'data', testset + '-' + kind + '.beancount')


    def load_test_data(self, testset, kind):
        filename = self.generate_file_name(testset, kind) 
        data, errors, _ = parser.parse_file(filename)
        assert not errors
        return data


if __name__ == '__main__':
    # show test case execution output iff logging level is DEBUG or finer:
    show_output = LOG_LEVEL <= logging.DEBUG
    unittest.main(buffer=not show_output)
