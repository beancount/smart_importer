"""
Decorator for a Beancount Importers's `extract` function
that suggests and predicts postings
using machine learning.
"""

import logging
from functools import wraps
from typing import List, Union

from beancount.core.data import Transaction, TxnPosting
from beancount.ingest.cache import _FileMemo
from beancount.ingest.importer import ImporterProtocol
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC

from smart_importer import machinelearning_helpers as ml

logger = logging.getLogger(__name__)


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
                 training_data: Union[_FileMemo, List[Transaction], str] = None,
                 filter_training_data_by_account: str = None,
                 predict_second_posting: bool = True,
                 suggest_accounts: bool = True):
        self.training_data = training_data
        self.filter_training_data_by_account = filter_training_data_by_account
        self.predict_second_posting = predict_second_posting
        self.suggest_accounts = suggest_accounts

    def __call__(self, importers_extract_function, *args, **kwargs):
        # Decorating the extract function:

        @wraps(importers_extract_function)
        def _extract(importerInstance: ImporterProtocol, *args, **kwargs) -> List[Transaction]:
            """
            Completes missing missing postings using machine learning.
            :param importerInstance: refers to the importer object, which is normally passed in
                as `self` argument.
            :param *args: original arguments to be passed
            :param **kwargs: original keyword arguments to be passed
            :return: list of beancount transactions
            """
            existing_entries = None
            if 'existing_entries' in kwargs:
                existing_entries = kwargs['existing_entries']


            # load training data
            self.training_data = ml.load_training_data(
                self.training_data,
                filter_training_data_by_account=self.filter_training_data_by_account,
                existing_entries=existing_entries)

            # convert training data to a list of TxnPostings
            self.converted_training_data = [TxnPosting(t, p) for t in self.training_data for p in t.postings
                                  # ...filtered, the TxnPosting.posting.account must be different from the
                                  # already-known filter_training_data_by_account and the reference account:
                                  if p.account != self.filter_training_data_by_account and p.account != t.postings[0].account]
            # train the machine learning model
            self._trained = False
            if not self.converted_training_data:
                logger.warning("Cannot train the machine learning model "
                               "because the training data is empty.")
            elif len(self.converted_training_data) < 2:
                logger.warning("Cannot train the machine learning model "
                               "because the training data consists of less than two elements.")
            else:
                transformers = []
                transformer_weights = {}
                transformers.append(
                    ('narration', Pipeline([
                        ('getNarration', ml.GetNarration()),
                        ('vect', CountVectorizer(ngram_range=(1, 3))),
                    ]))
                )
                transformer_weights['narration'] = 0.8
                transformers.append(
                    ('account', Pipeline([
                        ('getReferencePostingAccount', ml.GetReferencePostingAccount()),
                        ('vect', CountVectorizer(ngram_range=(1, 3))),
                    ]))
                )
                transformer_weights['account'] = 0.8

                distinctPayees = set(map(lambda trx: trx.txn.payee, self.converted_training_data))
                if len(distinctPayees) > 1:
                    transformers.append(
                        ('payee', Pipeline([
                            ('getPayee', ml.GetPayee()),
                            ('vect', CountVectorizer(ngram_range=(1, 3))),
                        ]))
                    )
                    transformer_weights['payee'] = 0.5

                transformers.append(
                    ('dayOfMonth', Pipeline([
                        ('getDayOfMonth', ml.GetDayOfMonth()),
                        ('caster', ml.ArrayCaster()),  # need for issue with data shape
                    ]))
                )
                transformer_weights['dayOfMonth'] = 0.1


                self.pipeline = Pipeline([
                    ('union', FeatureUnion(
                        transformer_list=transformers,
                        transformer_weights=transformer_weights)),
                    ('svc', SVC(kernel='linear')),
                ])
                logger.debug("About to train the machine learning model...")
                self.pipeline.fit(self.converted_training_data, ml.GetPostingAccount().transform(self.converted_training_data))
                logger.info("Finished training the machine learning model.")
                self._trained = True

            # import transactions by calling the importer's extract function
            logger.debug(f"About to call the importer's extract function in order to receive entries to be imported...")
            transactions: List[Transaction]
            transactions = importers_extract_function(importerInstance, *args, **kwargs)
            logger.debug(f"Received {len(transactions)} entries by calling the importer's extract function.")

            if not self._trained:
                logger.warning("Cannot generate predictions or suggestions "
                               "because there is no trained machine learning model.")
                return transactions

            # predict missing second postings
            if self.predict_second_posting:
                logger.debug("About to generate predictions for missing second postings...")
                predicted_accounts: List[str]
                predicted_accounts = self.pipeline.predict(transactions)
                transactions = [ml.add_posting_to_transaction(*t_a)
                                for t_a in zip(transactions, predicted_accounts)]
                logger.debug("Finished adding predicted accounts to the transactions to be imported.")

            # suggest accounts that are likely involved in the transaction
            if self.suggest_accounts:
                # get values from the SVC decision function
                logger.debug("About to generate suggestions about related accounts...")
                decision_values = self.pipeline.decision_function(transactions)

                # add a human-readable class label (i.e., account name) to each value, and sort by value:
                suggestions = [[account for _, account in sorted(list(zip(distance_values, self.pipeline.classes_)),
                                                                 key=lambda x: x[0], reverse=True)]
                               for distance_values in decision_values]

                # add the suggested accounts to each transaction:
                transactions = [ml.add_suggested_accounts_to_transaction(*t_s)
                                for t_s in zip(transactions, suggestions)]
                logger.debug("Finished adding suggested accounts to the transactions to be imported.")

            return transactions

        return _extract
