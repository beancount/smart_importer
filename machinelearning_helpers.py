"""Machine Learning Helpers."""
import json
import sys
from typing import Dict, Any, List, Union, Optional

import numpy as np
from beancount import loader
from beancount.core.data import Transaction, ALL_DIRECTIVES, Posting, TxnPosting
from beancount.ingest.cache import _FileMemo
from beancount.parser import printer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC


def transaction_involves_account(transaction: Transaction, account: Optional[str]) -> bool:
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
    Helper class to print data that is passed through a scikit-learn pipeline.
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
    Helper class for casting data into array shape.
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


class NoFitMixin:
    '''
    Mixin that helps implementing a custom scikit-learn transformer.
    This mixing implements a transformer's fit method that simply returns self.
    Compare https://signal-to-noise.xyz/post/sklearn-pipeline/
    '''

    def fit(self, X, y=None):
        return self


class GetPayee(TransformerMixin, NoFitMixin):
    '''
    Scikit-learn transformer to extract the payee.
    The input can be of type List[Transaction] or List[TxnPosting],
    the output is a List[str].
    '''
    def transform(self, data: Union[List[TxnPosting], List[Transaction]]):
        return [self._get_payee(d) for d in data]

    def _get_payee(self, d):
        if isinstance(d, Transaction):
            return d.payee
        elif isinstance(d, TxnPosting):
            return d.txn.payee


class GetNarration(TransformerMixin, NoFitMixin):
    '''
    Scikit-learn transformer to extract the narration.
    The input can be of type List[Transaction] or List[TxnPosting],
    the output is a List[str].
    '''
    def transform(self, data: Union[List[TxnPosting], List[Transaction]]):
        return [self._get_narration(d) for d in data]

    def _get_narration(self, d):
        if isinstance(d, Transaction):
            return d.narration
        elif isinstance(d, TxnPosting):
            return d.txn.narration


class GetPostingAccount(TransformerMixin, NoFitMixin):
    '''
    Scikit-learn transformer to extract the account name.
    The input can be of type List[Transaction] or List[TxnPosting].
    The account name is extracted from the last posting of each transaction,
    or from TxnPosting.posting.account of each TxnPosting.
    The output is a List[str].
    '''
    def transform(self, data: Union[List[TxnPosting], List[Transaction]]):
        return [self._get_posting_account(d) for d in data]

    def _get_posting_account(self, d):
        if isinstance(d, Transaction):
            return d.postings[-1].account
        elif isinstance(d, TxnPosting):
            return d.posting.account


class GetDayOfMonth(TransformerMixin, NoFitMixin):
    '''
    Scikit-learn transformer to extract the day of month when a transaction happened.
    The input can be of type List[Transaction] or List[TxnPosting],
    the output is a List[Date].
    '''
    def transform(self, data: Union[List[TxnPosting], List[Transaction]]):
        return [self._get_day_of_month(d) for d in data]

    def _get_day_of_month(self, d):
        if isinstance(d, Transaction):
            return d.date.day
        elif isinstance(d, TxnPosting):
            return d.txn.date.day
