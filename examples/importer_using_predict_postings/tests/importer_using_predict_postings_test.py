import logging
import unittest
from unittest.mock import Mock

from beancount.ingest.cache import _FileMemo
from beancount.parser import printer

from examples.importer_using_predict_postings.importer import Importer

LOG_LEVEL = logging.DEBUG
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


class PredictPostingsTest(unittest.TestCase):
    training_data = '''
    2017-11-20 * "Two Postings"
      Assets:Patrick:CHF  12 CHF
      Assets:Patrick:USD  12 CHF
    
    2017-11-20 * "Single Posting"
      Assets:Patrick:CHF  12 CHF
    '''

    def test_the_importer(self):
        logger.info("Running Test Case: {id}".format(id=self.id().split('.')[-1]))
        importer = Importer(['.*'])
        mocked_file = Mock(spec=_FileMemo)
        mocked_file.name = 'downloaded-transactions.csv'
        mocked_file.contents = Mock(name='contents', return_value='')
        entries = importer.extract(mocked_file, existing_entries=self.training_data)
        # printer.print_entries(entries)


if __name__ == '__main__':
    # show test case execution output iff logging level is DEBUG or finer:
    show_output = LOG_LEVEL <= logging.DEBUG
    unittest.main(buffer=not show_output)
