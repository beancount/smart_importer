"""
Decorator for a Beancount Importers's `extract` function
that suggests and predicts payees
using machine learning.
"""
import inspect
import logging
from typing import List, Union

from beancount.core.data import Transaction
from beancount.ingest.cache import _FileMemo
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC

from smart_importer import machinelearning_helpers as ml

logger = logging.getLogger(__name__)


class PredictPayees:
    '''
    Applying this decorator to a beancount importer
    will predict and auto-complete missing payees
    of the transactions to be imported.

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
                 filter_training_data_by_account: str = None,
                 predict_payees: bool = True,
                 overwrite_existing_payees=False,
                 suggest_payees: bool = True):
        self.training_data = training_data
        self.filter_training_data_by_account = filter_training_data_by_account
        self.predict_payees = predict_payees
        self.overwrite_existing_payees = overwrite_existing_payees
        self.suggest_payees = suggest_payees

    def __call__(self, OriginalImporter):
        decorator = self
        assert inspect.isclass(OriginalImporter), "The decorator can only be applied to classes (not functions)."

        class PredictPayeesImporter(OriginalImporter):
            def extract(self, file, existing_entries=None):
                logger.debug(f"About to call the importer's extract function to receive entries to be imported...")
                if 'existing_entries' in inspect.signature(super().extract).parameters:
                    decorator.imported_transactions = super().extract(file, existing_entries)
                else:
                    decorator.imported_transactions = super().extract(file)
                logger.debug(
                    f"Received {len(decorator.imported_transactions)} entries by calling the importer's extract function.")
                return decorator._extract(
                    file,
                    existing_entries
                )

        return PredictPayeesImporter

    def _extract(self, file, existing_entries) -> List[Transaction]:
        """Completes missing payees using machine learning."""

        # load training data
        self.training_data = ml.load_training_data(
            self.training_data,
            filter_training_data_by_account=self.filter_training_data_by_account,
            existing_entries=existing_entries)

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
            return self.imported_transactions

        # predict payees
        self.transactions = self.imported_transactions
        if self.predict_payees:
            logger.debug("About to generate predictions for payees...")
            predicted_payees: List[str]
            predicted_payees = self.pipeline.predict(self.imported_transactions)
            self.transactions = [ml.add_payee_to_transaction(*t_p, overwrite=self.overwrite_existing_payees)
                            for t_p in zip(self.imported_transactions, predicted_payees)]
            logger.debug("Finished adding predicted payees to the transactions to be imported.")

        # suggest likely payees
        if self.suggest_payees:
            # get values from the SVC decision function
            logger.debug("About to generate suggestions about likely payees...")
            decision_values = self.pipeline.decision_function(self.imported_transactions)

            # add a human-readable class label (i.e., payee's name) to each value, and sort by value:
            suggested_payees = [[payee for _, payee in sorted(list(zip(distance_values, self.pipeline.classes_)),
                                                              key=lambda x: x[0], reverse=True)]
                                for distance_values in decision_values]

            # add the suggested payees to each transaction:
            self.transactions = [ml.add_suggested_payees_to_transaction(*t_p)
                                 for t_p in zip(self.transactions, suggested_payees)]
            logger.debug("Finished adding suggested payees to the transactions to be imported.")

        return self.transactions
