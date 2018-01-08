"""
Decorator for beancount importers that adds smart prediction
and autocompletion of postings.
"""

from functools import wraps
from typing import Dict, List, Union

import numpy as np
from beancount.core.data import ALL_DIRECTIVES
from beancount.ingest.cache import _FileMemo
from beancount.ingest.importer import ImporterProtocol

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
                 filter_training_data_by_account: str = None,
                 predict_second_posting: bool = True,
                 suggest_accounts: bool = True,
                 debug: bool = False):
        # Handle arguments
        # print(inspect.stack()[0][3]) # prints the current function name
        # debug(**locals()) # see https://stackoverflow.com/a/9938156
        self.training_data = training_data
        self.filter_by_account = filter_training_data_by_account
        self.predict_second_posting = predict_second_posting
        self.suggest_accounts = suggest_accounts
        self.debug = debug

    def __call__(self, importers_extract_function, *args, **kwargs):
        # Decorating the extract function:

        @wraps(importers_extract_function)
        def _extract(importerInstance: ImporterProtocol, csvFile: _FileMemo) -> List[Union[ALL_DIRECTIVES]]:
            """
            Completes missing missing postings using machine learning.
            :param importerInstance: refers to the importer object, which is normally passed in
                as `self` argument.
            :param csvFile: `_FileMemo` of the csv file to be imported
            :return: list of beancount directives
            """

            # load the training data
            self._trained = False
            if isinstance(self.training_data, _FileMemo):
                print(f"Reading training data from {self.training_data.name}")
                x_train, y_train = ml.load_training_data_from_file(self.training_data,
                                                                   self.filter_by_account,
                                                                   debug=self.debug)
            else:
                print(f"Reading {len(self.training_data)} entries of training data")
                x_train, y_train = ml.load_training_data_from_entrylist(self.training_data,
                                                                        self.filter_by_account,
                                                                        debug=self.debug)

            # Define machine learning pipeline
            self.pipeline = ml.pipeline()

            # Train the machine learning model
            if not y_train:
                print("Warning: Cannot train the machine learning model because the training data is empty.")
            else:
                print(f"Training machine learning model...")
                self.pipeline.fit(x_train, y_train)
                self._trained = True

            # call the decorated extract function
            transactions: List[Union[ALL_DIRECTIVES]]
            transactions = importers_extract_function(importerInstance, csvFile)

            if not self._trained:
                print("Warning: Cannot predict postings because there is no trained machine learning model")
                return transactions

            # create data structures for scikit-learn
            transactions_scikit: Dict[str, np.ndarray]
            transactions_scikit, _ = ml.load_training_data_from_entrylist(transactions)

            # predict missing second postings
            if self.predict_second_posting:
                predicted_accounts: List[str]
                predicted_accounts = self.pipeline.predict(transactions_scikit)
                transactions = [ml.add_posting_to_transaction(*t_a)
                                for t_a in zip(transactions, predicted_accounts)]

            # suggest accounts that are likely involved in the transaction
            if self.suggest_accounts:
                # get values from the SVC decision function
                decision_values = self.pipeline.decision_function(transactions_scikit)

                # add a human-readable class label (i.e., account name) to each value, and sort by value:
                suggestions = [[account for _, account in sorted(list(zip(distance_values, self.pipeline.classes_)),
                           key=lambda x: x[0], reverse=True)]
                    for distance_values in decision_values]

                # add the suggested accounts to each transaction:
                transactions = [ml.add_suggestions_to_transaction(*t_s)
                                for t_s in zip(transactions, suggestions)]

            return transactions

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
