"""
Decorator for Beancount Importer classes
that suggests and predicts postings using machine learning.
"""
import inspect
import logging
from functools import wraps
from typing import List, Union

from beancount.core.data import Transaction, TxnPosting
from beancount.ingest.cache import _FileMemo
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC

from smart_importer import machinelearning_helpers as ml

# configure logging
LOG_LEVEL = logging.DEBUG
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


# the decorator
class PredictPostings:
    '''
    Applying this decorator to a beancount importer or its extract method
    will predict and auto-complete missing second postings
    of the transactions to be imported.

    Example:

    @PredictPostings(
        training_data="trainingdata.beancount",
        account="The:Importers:Already:Known:Accountname"
    )
    class MyImporter(ImporterProtocol):
        def extract(file):
          # do the import, return list of entries
    '''

    # Implementation notes for how to write decorators for classes, see e.g.,
    # https://stackoverflow.com/a/9910180
    # https://www.codementor.io/sheena/advanced-use-python-decorators-class-function-du107nxsv
    # https://andrefsp.wordpress.com/2012/08/23/writing-a-class-decorator-in-python/

    def __init__(
            self,
            *,
            training_data: Union[_FileMemo, List[Transaction], str] = None,
            account: str = None,
            predict_second_posting: bool = True,
            suggest_accounts: bool = True
    ):
        self.training_data = training_data
        self.account = account
        self.predict_second_posting = predict_second_posting
        self.suggest_accounts = suggest_accounts

    def __call__(self, to_be_decorated=None, *args, **kwargs):

        if inspect.isclass(to_be_decorated):
            logger.debug('The Decorator was applied to a class.')
            return self.patched_importer_class(to_be_decorated)

        elif inspect.isfunction(to_be_decorated):
            logger.debug('The Decorator was applied to an instancemethod.')
            return self.patched_extract_function(to_be_decorated)

    def patched_importer_class(self, importer_class):
        importer_class.extract = self.patched_extract_function(importer_class.extract)
        return importer_class

    def patched_extract_function(self, original_extract_function):
        decorator = self

        @wraps(original_extract_function)
        def wrapper(self, file, existing_entries=None):
            decorator.existing_entries = existing_entries

            logger.debug(f"About to call the importer's extract function to receive entries to be imported...")
            if 'existing_entries' in inspect.signature(original_extract_function).parameters:
                decorator.imported_transactions = original_extract_function(self, file, existing_entries)
            else:
                decorator.imported_transactions = original_extract_function(self, file)


            return decorator.enhance_transactions()

        return wrapper

    def enhance_transactions(self):# load training data
        self.training_data = ml.load_training_data(
            self.training_data,
            known_account=self.account,
            existing_entries=self.existing_entries)

        # convert training data to a list of TxnPostingAccounts
        self.converted_training_data = [ml.TxnPostingAccount(t, p, pRef.account)
                                        for t in self.training_data
                                        for pRef in t.postings
                                        for p in t.postings
                                        if p.account != pRef.account]

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
            self.pipeline.fit(self.converted_training_data,
                              ml.GetPostingAccount().transform(self.converted_training_data))
            logger.info("Finished training the machine learning model.")
            self._trained = True

        if not self._trained:
            logger.warning("Cannot generate predictions or suggestions "
                           "because there is no trained machine learning model.")
            return self.imported_transactions

        # predict missing second postings
        self.transactions = self.imported_transactions
        if self.predict_second_posting:
            logger.debug("About to generate predictions for missing second postings...")
            predicted_accounts: List[str]
            predicted_accounts = self.pipeline.predict(self.imported_transactions)
            self.transactions = [ml.add_posting_to_transaction(*t_a)
                                 for t_a in zip(self.transactions, predicted_accounts)]
            logger.debug("Finished adding predicted accounts to the transactions to be imported.")

        # suggest accounts that are likely involved in the transaction
        if self.suggest_accounts:
            # get values from the SVC decision function
            logger.debug("About to generate suggestions about related accounts...")
            decision_values = self.pipeline.decision_function(self.imported_transactions)

            # add a human-readable class label (i.e., account name) to each value, and sort by value:
            suggestions = [[account for _, account in sorted(list(zip(distance_values, self.pipeline.classes_)),
                                                             key=lambda x: x[0], reverse=True)]
                           for distance_values in decision_values]

            # add the suggested accounts to each transaction:
            self.transactions = [ml.add_suggested_accounts_to_transaction(*t_s)
                                 for t_s in zip(self.transactions, suggestions)]
            logger.debug("Finished adding suggested accounts to the transactions to be imported.")

        return self.transactions
