"""Machine Learning Helpers."""

import json
import logging
from typing import List, Union, Optional

import numpy as np
from typing import Tuple, NamedTuple
from beancount import loader
from beancount.core.data import Transaction, Posting, TxnPosting, filter_txns
from beancount.ingest.cache import _FileMemo
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


def load_training_data(training_data: Union[_FileMemo, List[Transaction], str],
                       known_account: str = None,
                       existing_entries: List[Tuple] = None) -> List[Transaction]:
    '''
    Loads training data
    :param training_data: The training data that shall be loaded.
        Can be provided as a string (the filename pointing to a beancount file),
        a _FileMemo instance,
        or a list of beancount entries
    :param known_account: Optional filter for the training data.
        If provided, the training data is filtered to only include transactions that involve the specified account.
    :param existing_entries: Optional existing entries to use instead of explicit training_data
    :return: Returns a list of beancount entries.
    '''
    if not training_data and existing_entries:
        logger.debug("Using existing entries for training data")
        training_data = list(filter_txns(existing_entries))
    elif isinstance(training_data, _FileMemo):
        logger.debug(f"Reading training data from _FileMemo \"{training_data.name}\"...")
        training_data, errors, _ = loader.load_file(training_data.name)
        assert not errors
        training_data = filter_txns(training_data)
    elif isinstance(training_data, str):
        logger.debug(f"Reading training data from file \"{training_data}\"...")
        training_data, errors, _ = loader.load_file(training_data)
        assert not errors
        training_data = filter_txns(training_data)
    logger.debug(f"Finished reading training data.")
    if known_account:
        training_data = [t for t in training_data
                         # ...filtered because the training data must involve the account:
                         if transaction_involves_account(t, known_account)]
        logger.debug(f"After filtering for account {known_account}, "
                     f"the training data consists of {len(training_data)} entries.")
    return training_data


def transaction_involves_account(transaction: Transaction, account: Optional[str]) -> bool:
    '''
    Returns whether a transactions involves a specific account,
    i.e., if any one of the transaction's postings uses the specified account name.
    '''
    if account is None:
        return True
    return any([posting.account == account for posting in transaction.postings])


def add_posting_to_transaction(transaction: Transaction, postings_account: str) -> Transaction:
    '''
    Adds a posting with specified postings_account to a transaction.
    '''

    ## implementation note:
    ## for how to modify transactions, see this code from beancount.core.interpolate.py:
    # new_postings = list(entry.postings)
    # new_postings.extend(get_residual_postings(residual, account_rounding))
    # entry = entry._replace(postings=new_postings)

    if len(transaction.postings) != 1:
        return transaction

    additionalPosting: Posting
    additionalPosting = Posting(postings_account, None, None, None, None, None)
    new_postings_list = list(transaction.postings)
    new_postings_list.extend([additionalPosting])
    transaction = transaction._replace(postings=new_postings_list)
    return transaction


def add_payee_to_transaction(transaction: Transaction, payee: str, overwrite=False) -> Transaction:
    '''
    Sets a transactions's payee.
    '''
    if not transaction.payee or overwrite:
        transaction = transaction._replace(payee=payee)
    return transaction


METADATA_KEY_SUGGESTED_ACCOUNTS = '__suggested_accounts__'
METADATA_KEY_SUGGESTED_PAYEES = '__suggested_payees__'


def add_suggested_accounts_to_transaction(transaction: Transaction, suggestions: List[str]) -> Transaction:
    """
    Adds suggested related accounts to a transaction.
    This function is a convenience wrapper over `_add_suggestions_to_transaction`.
    """
    return _add_suggestions_to_transaction(transaction, suggestions, key=METADATA_KEY_SUGGESTED_ACCOUNTS)


def add_suggested_payees_to_transaction(transaction: Transaction, suggestions: List[str]) -> Transaction:
    """
    Adds suggested payees to a transaction.
    This function is a convenience wrapper over `_add_suggestions_to_transaction`.
    """
    return _add_suggestions_to_transaction(transaction, suggestions, key=METADATA_KEY_SUGGESTED_PAYEES)


def _add_suggestions_to_transaction(transaction: Transaction, suggestions: List[str], key='__suggestions__'):
    """
    Adds a list of suggested accounts to a transaction under transaction.meta[key].
    """
    meta = transaction.meta
    meta[key] = json.dumps(suggestions)
    transaction = transaction._replace(meta=meta)
    return transaction

def merge_non_transaction_entries(imported_entries, enhanced_transactions):
    enhanced_entries = []
    enhanced_transactions_iter = iter(enhanced_transactions)
    for entry in imported_entries:
        if isinstance(entry, Transaction):
            enhanced_entries.append(next(enhanced_transactions_iter))
        else:
            enhanced_entries.append(entry)

    return enhanced_entries

TxnPostingAccount = NamedTuple('TxnPostingAccount', [
    ('txn', Transaction),
    ('posting', Posting),
    ('account', str)])

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
    The input can be of type List[Transaction] or List[TxnPostingAccount],
    the output is a List[str].
    '''

    def transform(self, data: Union[List[TxnPostingAccount], List[Transaction]]):
        return [self._get_payee(d) for d in data]

    def _get_payee(self, d):
        if isinstance(d, Transaction):
            return d.payee or ''
        elif isinstance(d, TxnPostingAccount):
            return d.txn.payee or ''


class GetNarration(TransformerMixin, NoFitMixin):
    '''
    Scikit-learn transformer to extract the narration.
    The input can be of type List[Transaction] or List[TxnPostingAccount],
    the output is a List[str].
    '''

    def transform(self, data: Union[List[TxnPostingAccount], List[Transaction]]):
        return [self._get_narration(d) for d in data]

    def _get_narration(self, d):
        if isinstance(d, Transaction):
            return d.narration
        elif isinstance(d, TxnPostingAccount):
            return d.txn.narration


class GetPostingAccount(TransformerMixin, NoFitMixin):
    '''
    Scikit-learn transformer to extract the account name.
    The input can be of type List[Transaction] or List[TxnPostingAccount].
    The account name is extracted from the last posting of each transaction,
    or from TxnPostingAccount.posting.account of each TxnPostingAccount.
    The output is a List[str].
    '''

    def transform(self, data: Union[List[TxnPosting], List[Transaction]]):
        return [self._get_posting_account(d) for d in data]

    def _get_posting_account(self, d):
        if isinstance(d, Transaction):
            return d.postings[-1].account
        elif isinstance(d, TxnPostingAccount):
            return d.posting.account

class GetReferencePostingAccount(TransformerMixin, NoFitMixin):
    '''
    Scikit-learn transformer to extract the reference account name.
    The input can be of type List[Transaction] or List[TxnPostingAccount].
    The reference account name is extracted from the first posting of each transaction.
    The output is a List[str].
    '''

    def transform(self, data: Union[List[TxnPostingAccount], List[Transaction]]):
        return [self._get_posting_account(d) for d in data]

    def _get_posting_account(self, d):
        if isinstance(d, Transaction):
            return d.postings[0].account
        elif isinstance(d, TxnPostingAccount):
            return d.account

class GetDayOfMonth(TransformerMixin, NoFitMixin):
    '''
    Scikit-learn transformer to extract the day of month when a transaction happened.
    The input can be of type List[Transaction] or List[TxnPostingAccount],
    the output is a List[Date].
    '''

    def transform(self, data: Union[List[TxnPostingAccount], List[Transaction]]):
        return [self._get_day_of_month(d) for d in data]

    def _get_day_of_month(self, d):
        if isinstance(d, Transaction):
            return d.date.day
        elif isinstance(d, TxnPostingAccount):
            return d.txn.date.day
