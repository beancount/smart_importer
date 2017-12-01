"""Smart CSV to Beancount Importer Implementation."""

import csv
import os
import unittest

from beancount.ingest import cache
from beancount.ingest.importers.csv import Col
from beancount.parser import printer
from beancount.utils import test_utils

from importers.smart_importer.smart_csv_importer import SmartCsvImporter


class SmartCsvImporterTest(unittest.TestCase):
    def test_extract(self):
        self.fail()

    @test_utils.docfile
    def test_extract(self, filename):
        # pylint: disable=line-too-long
        """\
        DE40100100100000012345;Paying the rent;03.05.2016;03.05.2016;-2400,00;EUR
        DE40100100100000012345;Monthly bank fee;04.05.2016;05.09.2017;-4,00;EUR
        DE40100100100000012345;Wine-Tarner Cable;25.06.2016;25.06.2016;-12,10;EUR
        """
        file = cache.get_file(filename)

        class ExampleCsvDialect(csv.Dialect):
            lineterminator = '\n'
            delimiter = ';'
            quotechar = '"'
            quoting = csv.QUOTE_MINIMAL

        importer = SmartCsvImporter(
            # config:
            {
                # Col.IBAN: first col is the account's iban, which we ignore
                Col.NARRATION1: 1,
                Col.DATE: 2,
                Col.TXN_DATE: 3,
                Col.AMOUNT: 4
                # Col.CURRENCY: last col is the currency, which we ignore
            },
            # account:
            'Assets:US:BofA:Checking',
            # currency:
            'USD',
            # regexps:
            ('Details,Posting Date,"Description",Amount,'
             'Type,Balance,Check or Slip #,'),
            # beancount file to learn from:
            beancount_file=cache.get_file(
                os.path.abspath(os.path.join(os.path.dirname(__file__), 'training_data.beancount')),
            ),
            debug=False,
            csv_dialect=ExampleCsvDialect()
        )

        entries = importer.extract(file)

        print('Entries with predicted accounts:')
        printer.print_entries(entries)

        expected_results = {
            0: 'Expenses:Home:Rent',
            1: 'Expenses:Financial:Fees',
            2: 'Expenses:Home:Internet'
        }

        for index, expected_account in expected_results.items():
            predicted_account = entries[index].postings[-1].account
            self.assertEqual(predicted_account, expected_account,
                             'Predicted account for "{narration}" is "{predicted_account}", but should be "{expected_account}".'.format(
                                 narration=entries[index].narration,
                                 predicted_account=predicted_account,
                                 expected_account=expected_account
                             ))
