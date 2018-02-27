import logging
import unittest
from unittest.mock import Mock

from beancount.core import data, amount
from beancount.core.number import D
from beancount.ingest import importer
from beancount.ingest.cache import _FileMemo
from beancount.ingest.importers import regexp
from beancount.parser import printer
from dateutil.parser import parse

from smart_importer.predict_postings import PredictPostings

LOG_LEVEL = logging.DEBUG
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


@PredictPostings()
class Importer(regexp.RegexpImporterMixin, importer.ImporterProtocol):
    """An importer for account statements."""

    def __init__(self, regexps):
        if isinstance(regexps, str):
            regexps = [regexps]
        regexp.RegexpImporterMixin.__init__(self, regexps)

    def file_account(self, file):
        return 'Assets:Foo'

    def extract(self, file, existing_entries=None):
        entries = []
        meta = data.new_metadata(file.name, 0)
        txn = data.Transaction(
            meta,
            parse('2017-11-20').date(),
            '*',
            None,
            'Two Postings',
            data.EMPTY_SET,
            data.EMPTY_SET,
            [
                data.Posting('Assets:Patrick:CHF', amount.Amount(D('12'), 'CHF'), None, None, None, None),
                data.Posting('Assets:Patrick:USD', amount.Amount(D('12'), 'CHF'), None, None, None, None),
            ]
        )

        entries.append(txn)
        txn = data.Transaction(
            meta,
            parse('2017-11-20').date(),
            '*',
            None,
            'Single Posting',
            data.EMPTY_SET,
            data.EMPTY_SET,
            [
                data.Posting('Assets:Patrick:CHF', amount.Amount(D('12'), 'CHF'), None, None, None, None),
            ]
        )

        entries.append(txn)

        return entries


class PredictPostingsTest(unittest.TestCase):

    training_data = '''
    2017-11-20 * "Two Postings"
      Assets:Patrick:CHF  12 CHF
      Assets:Patrick:USD  12 CHF
    
    2017-11-20 * "Single Posting"
      Assets:Patrick:CHF  12 CHF
    '''

    def test_the_importer(self):
        importer = Importer(['.*'])
        mocked_file = Mock(spec=_FileMemo)
        mocked_file.name = 'downloaded-transactions.csv'
        mocked_file.contents = Mock(name='contents', return_value='')
        entries = importer.extract(mocked_file, existing_entries=self.training_data)
        printer.print_entries(entries)


if __name__ == '__main__':
    # show test case execution output iff logging level is DEBUG or finer:
    show_output = LOG_LEVEL <= logging.DEBUG
    unittest.main(buffer=not show_output)
