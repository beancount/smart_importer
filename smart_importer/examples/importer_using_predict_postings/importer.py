'''
Example importer that uses the PredictPostings decorator.
'''

from beancount.core import data, amount
from beancount.core.number import D
from beancount.ingest import importer
from beancount.ingest.importers import regexp
from dateutil.parser import parse

from smart_importer.predict_postings import PredictPostings


@PredictPostings()
class Importer(regexp.RegexpImporterMixin, importer.ImporterProtocol):
    """An importer for account statements."""

    def __init__(self, regexps):
        if isinstance(regexps, str):
            regexps = [regexps]
        regexp.RegexpImporterMixin.__init__(self, regexps)

    def file_account(self, file):
        return 'Assets:Foo'

    def extract(self, file, existing_entries):
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
