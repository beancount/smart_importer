"""
Decorator for beancount importers that adds smart prediction
and autocompletion of postings.
"""

from functools import wraps
from typing import Dict, List, Union

import numpy as np
from beancount.core.data import ALL_DIRECTIVES
from beancount.ingest.cache import _FileMemo

import importers.smart_importer.machinelearning as ml


class PredictPostings:
    '''
    Applying this decorator to the extract function of a beancount importer
    will predict and auto-complete missing second postings
    of the transactions that are imported.

    Predictions are implemented using machine learning
    based on training data read from a beancount file.

    Example:

    class MyImporter(ImporterProtocol):
        @PredictPostings(training_data="trainingdata.beancount")
        def extract(file):
          # do the import, return list of entries
    '''

    # Implementation notes for how to write class-based decorators,
    # see http://scottlobdell.me/2015/04/decorators-arguments-python/

    def __init__(self, *,
                 training_data: Union[_FileMemo, List[Union[ALL_DIRECTIVES]]],
                 filter_by_account: str = None,
                 debug: bool = False):
        # Handle arguments
        # print(inspect.stack()[0][3]) # prints the current function name
        # debug(**locals()) # see https://stackoverflow.com/a/9938156
        self._trained = False

        # load the training data
        if isinstance(training_data, _FileMemo):
            print(f"Reading training data from {training_data.name}")
            x_train, y_train = ml.load_training_data_from_file(training_data, filter_by_account, debug=debug)
        else:
            print(f"Reading {len(training_data)} entries of training data")
            x_train, y_train = ml.load_training_data_from_entrylist(training_data, filter_by_account, debug=debug)

        # Define machine learning pipeline
        self.pipeline = ml.pipeline()

        # Train the machine learning model
        if not y_train:
            print("Warning: Cannot train the machine learning model because the training data is empty.")
        else:
            print(f"Training machine learning model...")
            self.pipeline.fit(x_train, y_train)
            self._trained = True

    def __call__(self, importers_extract_function, *args, **kwargs):
        # Decorating the extract function:

        @wraps(importers_extract_function)
        def _extract(importerInstance, csvFile: _FileMemo) -> List[Union[ALL_DIRECTIVES]]:
            """
            Completes missing missing postings using machine learning.
            :param importerInstance: refers to the importer object, which is normally passed in
                as `self` argument.
            :param csvFile: `_FileMemo` of the csv file to be imported
            :return: list of beancount directives
            """

            # use the importer to import the file
            transactions_beancount: List[Union[ALL_DIRECTIVES]]
            transactions_beancount = importers_extract_function(importerInstance, csvFile)

            if not self._trained:
                print("Warning: Cannot predict postings because there is no trained machine learning model")
                return transactions_beancount

            # create data structures for scikit-learn
            transactions_scikit: Dict[str, np.ndarray]
            transactions_scikit, _ = ml.load_training_data_from_entrylist(transactions_beancount)

            # predict missing postings
            predicted_accounts: List[str]
            predicted_accounts = self.pipeline.predict(transactions_scikit)
            transactions_with_predicted_accounts = [ml.add_account_to_transaction(*t_a) for t_a in
                                                    zip(transactions_beancount, predicted_accounts)]
            return transactions_with_predicted_accounts

        return _extract

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
