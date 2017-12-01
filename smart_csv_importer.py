"""Smart CSV to Beancount Importer Implementation."""

import csv
import os
from typing import List, Union, Dict

import beancount
import numpy as np
from beancount.core.data import ALL_DIRECTIVES
from beancount.ingest import cache
from beancount.ingest import regression
from beancount.ingest.cache import _FileMemo
from beancount.ingest.importers.csv import Col

import importers.smart_importer.machinelearning as ml


class SmartCsvImporter(beancount.ingest.importers.csv.Importer):
    """
    A smart CSV importer implementation.

    The importer converts a CSV file to beancount directives,
    and then uses machine learning to predict missing payees and missing postings.
    The extraction functionality is implemented using `beancount.ingest.importers.csv.Importer`.
    The autocompletion functionality is implemented in this class using the scikit-learn machine learning library.
    """

    # Todo, potential refactorings of the Smart CSV Importer:
    # What should the data format look like for returning predictions?
    # * We could extend the beancount.core.data directives with additional attributes,
    #   see https://stackoverflow.com/a/45193692 for how to extend named tuples.
    # * Or we could use the metadata field?
    # * How should we specify what data was imported vs what data has been completed through machine learning?
    # * How should we return ranked suggestions for populating a dropdown list with likely account names?


    def __init__(self, config, account, currency, regexps, beancount_file: _FileMemo, institution=None, debug=False,
                 csv_dialect: Union[str, csv.Dialect] = 'excel'):
        """
        Instantiates the SmartCsvImporter.
        :param beancount_file: The beancount file with training data, instantiated, e.g., using `beancount.ingest.cache.get_file`
        """
        super().__init__(config, account, currency, regexps, institution=institution, debug=debug,
                         csv_dialect=csv_dialect)
        self.beancount_file = beancount_file
        self.pipeline = ml.pipeline()
        self._trained = False

    def extract(self, csvFile: _FileMemo) -> List[Union[ALL_DIRECTIVES]]:
        """
        Converts CSV to beancount directives and completes missing payees and missing postings using machine learning.
        :param csvFile: csv file to be imported
        :return: list of beancount directives
        """

        # learn from existing beancount data:
        if not self._trained:
            self._train()
        # note: the _trained model now is in `self.pipeline`.

        # read CSV file
        transactions_beancount: List[Union[ALL_DIRECTIVES]]
        transactions_beancount = super().extract(csvFile)
        transactions_scikit, _ = ml.load_training_data_from_entrylist(transactions_beancount)
        predicted_accounts = self.pipeline.predict(transactions_scikit)

        transactions_with_predicted_accounts = [ml.add_account_to_transaction(*t_a) for t_a in
                                                zip(transactions_beancount, predicted_accounts)]
        return transactions_with_predicted_accounts

    def _train(self):
        '''
        Trains a machine learning model from `self.beancount_file`.
        '''
        # load the beancount file
        x_train, y_train = ml.load_training_data_from_file(self.beancount_file, self.account)

        # _train the machine learning model
        self.pipeline.fit(x_train, y_train)
        self._trained = True

    def _predict(self, transactions_dict: Dict[str, np.ndarray]) -> List[str]:
        predicted_accounts = self.pipeline.predict(transactions_dict)
        return predicted_accounts


def test():
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
        debug=True,
        csv_dialect=ExampleCsvDialect()
    )

    test_dir = os.path.join(os.path.basename(os.path.dirname(os.path.realpath(__file__))), 'tests')
    print(test_dir)
    yield from regression.compare_sample_files(importer, directory=test_dir)


if __name__ == "__main__":
    test()
