"""
Decorator for a Beancount Importers's `extract` function
that suggests and predicts payees
using machine learning.
"""
import inspect
import logging
from functools import wraps
from typing import List, Union

from beancount.core.data import Transaction, filter_txns
from beancount.ingest.cache import _FileMemo
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC

from smart_importer import machinelearning_helpers as ml

# configure logging
from smart_importer.decorator_baseclass import SmartImporterDecorator

LOG_LEVEL = logging.DEBUG
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


class PredictPayees(SmartImporterDecorator):
    '''
    Applying this decorator to a beancount importer or its extract method
    will predict and auto-complete missing payees
    of the transactions that are imported.

    Predictions are implemented using machine learning
    based on training data read from a beancount file.

    Example:

    @PredictPayees(training_data="trainingdata.beancount")
    class MyImporter(ImporterProtocol):
        def extract(file):
          # do the import, return list of entries
    '''

    # Implementation notes for how to write class-based decorators,
    # see http://scottlobdell.me/2015/04/decorators-arguments-python/

    def __init__(self, *,
                 training_data: Union[_FileMemo, List[Transaction], str] = None,
                 account: str = None,
                 predict_payees: bool = True,
                 overwrite_existing_payees=False,
                 suggest_payees: bool = True):
        self.training_data = training_data
        self.account = account
        self.predict_payees = predict_payees
        self.overwrite_existing_payees = overwrite_existing_payees
        self.suggest_payees = suggest_payees

    def main(self):
        '''
        Adds predicted payees to the transactions.
        '''
        # load training data
        self.training_data = ml.load_training_data(
            self.training_data,
            known_account=self.account,
            existing_entries=self.existing_entries)

        # train the machine learning model
        self._trained = False
        if not self.training_data:
            logger.warning("Cannot train the machine learning model "
                           "because the training data is empty.")
        elif len(self.training_data) < 2:
            logger.warning("Cannot train the machine learning model "
                           "because the training data consists of less than two elements.")
        else:
            self.pipeline = Pipeline([
                ('union', FeatureUnion(
                    transformer_list=[
                        ('narration', Pipeline([
                            ('getNarration', ml.GetNarration()),
                            ('vect', CountVectorizer(ngram_range=(1, 3))),
                        ])),
                        ('payee', Pipeline([  # any existing payee, if one exists
                            ('getPayee', ml.GetPayee()),
                            ('vect', CountVectorizer(ngram_range=(1, 3))),
                        ])),
                        ('dayOfMonth', Pipeline([
                            ('getDayOfMonth', ml.GetDayOfMonth()),
                            ('caster', ml.ArrayCaster()),  # need for issue with data shape
                        ])),
                    ],
                    transformer_weights={
                        'narration': 0.8,
                        'payee': 0.5,
                        'dayOfMonth': 0.1
                    })),
                ('svc', SVC(kernel='linear')),
            ])
            logger.debug("About to train the machine learning model...")
            self.pipeline.fit(self.training_data, ml.GetPayee().transform(self.training_data))
            logger.info("Finished training the machine learning model.")
            self._trained = True

        if not self._trained:
            logger.warning("Cannot generate predictions or suggestions "
                           "because there is no trained machine learning model.")
            return self.imported_entries

        imported_transactions: List[Transaction]
        imported_transactions = list(filter_txns(self.imported_entries))
        enhanced_transactions: List[Transaction]
        enhanced_transactions = list(imported_transactions)

        # predict payees
        if self.predict_payees:
            logger.debug("About to generate predictions for payees...")
            predicted_payees: List[str]
            predicted_payees = self.pipeline.predict(enhanced_transactions)
            enhanced_transactions = [ml.add_payee_to_transaction(*t_p, overwrite=self.overwrite_existing_payees)
                                     for t_p in zip(enhanced_transactions, predicted_payees)]
            logger.debug("Finished adding predicted payees to the transactions to be imported.")

        # suggest likely payees
        if self.suggest_payees:
            # get values from the SVC decision function
            logger.debug("About to generate suggestions about likely payees...")
            decision_values = self.pipeline.decision_function(enhanced_transactions)

            # add a human-readable class label (i.e., payee's name) to each value, and sort by value:
            suggested_payees = [[payee for _, payee in sorted(list(zip(distance_values, self.pipeline.classes_)),
                                                              key=lambda x: x[0], reverse=True)]
                                for distance_values in decision_values]

            # add the suggested payees to each transaction:
            enhanced_transactions = [ml.add_suggested_payees_to_transaction(*t_p)
                                     for t_p in zip(enhanced_transactions, suggested_payees)]
            logger.debug("Finished adding suggested payees to the transactions to be imported.")

        return ml.merge_non_transaction_entries(self.imported_entries, enhanced_transactions)
