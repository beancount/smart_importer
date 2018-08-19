"""
Decorator for a Beancount Importers's `extract` function
that suggests and predicts payees
using machine learning.
"""
import logging
from typing import List, Union

from beancount.core.data import Transaction, filter_txns, ALL_DIRECTIVES
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
    will predict and suggest payees
    of imported transactions.

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
        Predicts and suggests payees for imported transactions.
        '''
        try:
            self.load_training_data()
            self.prepare_training_data()
            self.define_pipeline()
            self.train_pipeline()
            return self.process_entries()
        except (ValueError, AssertionError) as e:
            logger.error(e)
            return self.imported_entries

    def load_training_data(self):
        '''
        Loads training data, i.e., a list of beancount entries.
        '''
        self.training_data = ml.load_training_data(
            self.training_data,
            known_account=self.account,
            existing_entries=self.existing_entries)

    def prepare_training_data(self):
        '''
        Prepares the training data in preparation for training the machine learning pipeline.
        In the case of this decorator, no steps are necessary here.
        '''

        pass

    def define_pipeline(self):
        '''
        Defines the machine learning pipeline for predicting and suggesting payees.
        '''
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

    def train_pipeline(self):
        '''
        Trains the machine learning pipeline.
        '''
        if not self.training_data:
            raise ValueError("Cannot train the machine learning model "
                             "because the training data is empty.")
        elif len(self.training_data) < 2:
            raise ValueError("Cannot train the machine learning model "
                             "because the training data consists of less than two elements.")
        logger.debug("About to train the machine learning model...")
        self.pipeline.fit(self.training_data, ml.GetPayee().transform(self.training_data))
        logger.info("Finished training the machine learning model.")

    def process_entries(self) -> List[Union[ALL_DIRECTIVES]]:
        '''
        Processes all imported entries (transactions as well as other types of entries).
        Transactions are enhanced, but all other entries are left as is.
        :return: Returns the list of entries to be imported.
        '''
        imported_transactions: List[Transaction]
        imported_transactions = list(filter_txns(self.imported_entries))
        enhanced_transactions = self.process_transactions(list(imported_transactions))
        return ml.merge_non_transaction_entries(self.imported_entries, enhanced_transactions)

    def process_transactions(self, transactions: List[Transaction]) -> List[Transaction]:
        '''
        Processes all imported transactions:
        * Predicts payees
        * Suggests payees that are likely also involved in the transaction
        :param transactions: List of beancount transactions
        :return: List of beancount transactions
        '''

        # predict payees
        if self.predict_payees:
            logger.debug("About to generate predictions for payees...")
            predicted_payees: List[str]
            predicted_payees = self.pipeline.predict(transactions)
            transactions = [ml.add_payee_to_transaction(*t_p, overwrite=self.overwrite_existing_payees)
                            for t_p in zip(transactions, predicted_payees)]
            logger.debug("Finished adding predicted payees to the transactions to be imported.")

        # suggest likely payees
        if self.suggest_payees:
            # get values from the SVC decision function
            logger.debug("About to generate suggestions about likely payees...")
            decision_values = self.pipeline.decision_function(transactions)

            # add a human-readable class label (i.e., payee's name) to each value, and sort by value:
            suggested_payees = [[payee for _, payee in sorted(list(zip(distance_values, self.pipeline.classes_)),
                                                              key=lambda x: x[0], reverse=True)]
                                for distance_values in decision_values]

            # add the suggested payees to each transaction:
            transactions = [ml.add_suggested_payees_to_transaction(*t_p)
                            for t_p in zip(transactions, suggested_payees)]
            logger.debug("Finished adding suggested payees to the transactions to be imported.")
        return transactions
