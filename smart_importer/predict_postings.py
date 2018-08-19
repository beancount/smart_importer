"""
Decorator for Beancount Importer classes
that suggests and predicts postings using machine learning.
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


class PredictPostings(SmartImporterDecorator):
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
        self._trained = False

    def main(self):
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
        self.training_data = ml.load_training_data(
            self.training_data,
            known_account=self.account,
            existing_entries=self.existing_entries)

    def prepare_training_data(self):
        self.converted_training_data = [ml.TxnPostingAccount(t, p, pRef.account)
                                        for t in self.training_data
                                        for pRef in t.postings
                                        for p in t.postings
                                        if p.account != pRef.account]

    def define_pipeline(self):
        if not self.converted_training_data:
            raise ValueError("Cannot define the machine learning pipeline "
                             "because the converted training data is empty")

        if len(self.converted_training_data) < 2:
            raise ValueError("Cannot define the machine learning pipeline "
                             "because the training data consists of less than two elements.")

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

    def train_pipeline(self):
        if not self.converted_training_data:
            raise ValueError("Cannot train the machine learning model "
                             "because the converted training data is empty")

        if len(self.converted_training_data) < 2:
            raise ValueError("Cannot train the machine learning model "
                             "because the training data consists of less than two elements.")

        self.pipeline.fit(self.converted_training_data,
                          ml.GetPostingAccount().transform(self.converted_training_data))
        logger.info("Finished training the machine learning model.")

    def process_entries(self) -> List[Union[ALL_DIRECTIVES]]:
        '''
        Processes all imported entries (transactions and others).
        All transactions are enhanced, all other entries are left as is.
        :return: Returns the list of entries to be imported.
        '''
        imported_transactions: List[Transaction]
        imported_transactions = list(filter_txns(self.imported_entries))
        enhanced_transactions = self.process_transactions(list(imported_transactions))
        return ml.merge_non_transaction_entries(self.imported_entries, enhanced_transactions)

    def process_transactions(self, transactions: List[Transaction]) -> List[Transaction]:
        '''
        Processes all imported transactions:
        * Predicts missing second postings.
        * Suggests accounts that are likely also involved in the transaction
        :param transactions: List of beancount transactions
        :return: List of beancount transactions
        '''
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
