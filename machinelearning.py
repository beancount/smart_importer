"""Machinelearning Helpers for the Smart Importer."""
import json
import sys
from typing import Dict, Any, List, Union, Optional

import numpy as np
from beancount import loader
from beancount.core.data import Transaction, ALL_DIRECTIVES, Posting
from beancount.ingest.cache import _FileMemo
from beancount.parser import printer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC


def load_training_data_from_file(beancount_file: _FileMemo, filter_by_account: str, debug=False) -> (
        Dict[str, np.ndarray], List[str]):
    """
    Loads entries from a beancount file and returns them in a format suitable
    for training a scikit-learn machine learning model.
    :param beancount_file:
    :param filter_by_account:
    :param debug:
    :return:
    """

    # load the beancount file
    entries: List[Union[ALL_DIRECTIVES]]
    entries, errors, option_map = loader.load_file(beancount_file.name)

    if errors:
        print(',----------------------------------------------------------------------')
        printer.print_errors(errors, file=sys.stdout)
        print('`----------------------------------------------------------------------')
        raise ValueError('Error loading beancount file {}'.format(beancount_file.name))

    if debug:
        print('Loaded {} entries from the beancount file.'.format(len(entries)))
        # printer.print_entries(entries)

    # filter directives to only include transactions that involve the given account
    transaction_entries: List[Transaction]
    transaction_entries = [entry for entry in entries
                           if isinstance(entry, Transaction)
                           and transaction_involves_account(entry, filter_by_account)]
    if debug:
        print('Identified {} transactions relevant to {}.'.format(len(transaction_entries), filter_by_account))
        # printer.print_entries(transaction_entries)

    transactions_dict, accounts = load_training_data_from_entrylist(transaction_entries,
                                                                    filter_by_account=filter_by_account,
                                                                    debug=debug)

    return transactions_dict, accounts


def load_training_data_from_entrylist(
        transaction_entries: List[Union[ALL_DIRECTIVES]],
        filter_by_account: Optional[str] = None,
        debug: bool = False) \
        -> (Dict[str, np.ndarray], List[str]):
    """
    converts a list of beancount entries into a format suitable
    for training a scikit-learn machine learning model.
    :param transaction_entries: list of beancount entries. any entries that are not transactions are discarded.
    :param filter_by_account: postings with this account name will be ignored in the output
    :param debug: set to True in order to print debug statements
    :return: tupel of: transactions (returned as one numpy array per feature), and accounts (array of strings).
    """

    # discard any entries that are not transactions
    transaction_entries = [entry for entry in transaction_entries
                           if isinstance(entry, Transaction)]

    # flatten the list of transactions with nested postings into a flat list of postings
    # TODO: Refactor, use the TxnPosting class?
    transactions: Dict[str, Any]
    accounts: List[str]
    if not transaction_entries:
        transactions = []
        accounts = []
    else:
        transactions, accounts = zip(*[
            ({
                 'date_day': transaction.date.day,
                 'date_month': transaction.date.month,
                 'payee': transaction.payee,
                 'narration': transaction.narration
             }, posting.account)
            for transaction in transaction_entries
            for posting in transaction.postings
            if posting.account != filter_by_account or filter_by_account is None])

    # prepare a data structure as needed by scikit-learn, i.e.,
    # transactions_dict is of type { 'variable': [ 'list', 'of, 'values'], 'variable2': ['list', 'of', 'values']}
    keys = ['narration', 'date_day', 'date_month']
    transactions_dict: Dict[str, np.ndarray]
    transactions_dict = {k: np.array([t[k] for t in transactions]) for k in keys}

    if debug and len(transactions_dict['narration']):
        print("Prepared data for machine learning. Example transaction and filter_by_account:")
        i = 0
        print({key: transactions_dict[key][i] for key in keys}, {'filter_by_account', accounts[i]})
    return transactions_dict, accounts


def transaction_involves_account(transaction: Transaction, account: str) -> bool:
    '''
    Returns whether a transactions involves a specific account,
    i.e., if any one of the transaction's postings uses the specified account name.
    '''
    if account is None:
        return True
    return any([posting.account == account for posting in transaction.postings])


def add_posting_to_transaction(transaction: Transaction, postings_account: str):
    '''
    Modifies a transaction by adding a posting with specified postings_account to it.
    '''

    ## implementation note:
    ## for how to modify transactions, see this code from beancount.core.interpolate.py:
    # new_postings = list(entry.postings)
    # new_postings.extend(get_residual_postings(residual, account_rounding))
    # entry = entry._replace(postings=new_postings)

    additionalPosting: Posting
    additionalPosting = Posting(postings_account, None, None, None, None, None)
    new_postings_list = list(transaction.postings)
    new_postings_list.extend([additionalPosting])
    transaction = transaction._replace(postings=new_postings_list)
    return transaction


def add_suggestions_to_transaction(transaction: Transaction, suggested_accounts: List[str]):
    """
    Adds a list of suggested accounts to a transaction under transaction.meta['__suggested_accounts__'].
    :param transaction:
    :param suggested_accounts:
    :return:
    """
    meta = transaction.meta
    meta['__suggested_accounts__'] = json.dumps(suggested_accounts)
    transaction = transaction._replace(meta=meta)
    return transaction


def pipeline() -> Pipeline:
    """
    Returns a scikit-learn machine learning pipeline for predicting account names.
    """
    return Pipeline([
        # Use FeatureUnion to combine the features from subject and body
        ('union', FeatureUnion(
            transformer_list=[

                # Pipeline for narration
                ('narrationA', Pipeline([
                    ('selector', ItemSelector(key='narration')),
                    #                 ('tfidf', TfidfVectorizer(min_df=2)),
                    ('countvectorizer', CountVectorizer(ngram_range=(1, 3))),
                    #                 ('printer', StatusPrinter()),
                    #                 ('logisticregression', LogisticRegression()),
                ])),

                # Pipeline day
                ('date_dayA', Pipeline([
                    ('selector', ItemSelector(key='date_day')),
                    #                ('svc', SVC()),
                    ('caster', ArrayCaster())  # need for issues with data shape
                ])),

                # Pipeline month (DAY AND MONTH SHOULD BE IN ONE ESTIMATOR)
                ('date_month', Pipeline([
                    ('selector', ItemSelector(key='date_month')),
                    #                ('svc', SVC()),
                    ('caster', ArrayCaster())
                ])),

            ],

            # weight components in FeatureUnion
            transformer_weights={
                'narration': 0.8,
                'date_day': 0.5,
                'date_month': 0.1,
            },
        )),

        # Use a SVC classifier on the combined features
        ('svc', SVC(kernel='linear')),
    ])


class ItemSelector(BaseEstimator, TransformerMixin):
    """
    Helper class:
    For data grouped by feature, select subset of data at a provided key.

    Code from:
    http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to scikit-learn feature
    matrixes (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


class StatusPrinter(BaseEstimator, TransformerMixin):
    """
    Helper class.
    """

    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        print(data)
        return data


class ArrayCaster(BaseEstimator, TransformerMixin):
    """
    Helper class.
    """

    def __init__(self, debug=False):
        self.debug = debug

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        if self.debug:
            print(data.shape)
            print(np.transpose(np.matrix(data)).shape)
        return np.transpose(np.matrix(data))
