"""Machine Learning Helpers."""

import logging
from typing import List, Union, Tuple, NamedTuple

import numpy
from beancount import loader
from beancount.core.data import Transaction, Posting, TxnPosting, filter_txns
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


def load_training_data(
        training_data: Union[List[Transaction], str],
        known_account: str = None,
        existing_entries: List[Tuple] = None) -> List[Transaction]:
    """Load training data.

    :param training_data: The training data that shall be loaded.
        Can be provided as a string (the filename pointing to a beancount
            file),
        or a list of Beancount entries
    :param known_account: Optional filter for the training data.
        If provided, the training data is filtered to only include transactions
        that involve the specified account.
    :param existing_entries: Optional existing entries to use instead of
        explicit training_data
    :return: A list of Beancount entries.
    """
    if not training_data and existing_entries:
        logger.debug("Using existing entries for training data")
        training_data = existing_entries
    elif isinstance(training_data, str):
        logger.debug(f"Reading training data from file \"{training_data}\".")
        training_data, _, __ = loader.load_file(training_data)
    logger.debug("Finished reading training data.")
    if training_data:
        training_data = list(filter_txns(training_data))
    if known_account:
        training_data = [
            txn for txn in training_data
            if any([pos.account == known_account for pos in txn.postings])
        ]
        logger.debug(
            f"After filtering for account {known_account}, "
            f"the training data consists of {len(training_data)} entries.")
    return training_data


TxnPostingAccount = NamedTuple('TxnPostingAccount',
                               [('txn', Transaction), ('posting', Posting),
                                ('account', str)])


class ArrayCaster(BaseEstimator, TransformerMixin):
    """
    Helper class for casting data into array shape.
    """

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        return numpy.transpose(numpy.matrix(data))


class NoFitMixin:
    """
    Mixin that helps implementing a custom scikit-learn transformer.
    This mixing implements a transformer's fit method that simply returns self.
    Compare https://signal-to-noise.xyz/post/sklearn-pipeline/
    """

    def fit(self, X, y=None):
        return self


class GetPayee(TransformerMixin, NoFitMixin):
    """
    Scikit-learn transformer to extract the payee.
    The input can be of type List[Transaction] or List[TxnPostingAccount],
    the output is a List[str].
    """

    def transform(self,
                  data: Union[List[TxnPostingAccount], List[Transaction]]):
        return [self._get_payee(d) for d in data]

    def _get_payee(self, d):
        if isinstance(d, Transaction):
            return d.payee or ''
        elif isinstance(d, TxnPostingAccount):
            return d.txn.payee or ''


class GetNarration(TransformerMixin, NoFitMixin):
    """
    Scikit-learn transformer to extract the narration.
    The input can be of type List[Transaction] or List[TxnPostingAccount],
    the output is a List[str].
    """

    def transform(self,
                  data: Union[List[TxnPostingAccount], List[Transaction]]):
        return [self._get_narration(d) for d in data]

    def _get_narration(self, d):
        if isinstance(d, Transaction):
            return d.narration
        elif isinstance(d, TxnPostingAccount):
            return d.txn.narration


class GetPostingAccount(TransformerMixin, NoFitMixin):
    """
    Scikit-learn transformer to extract the account name.
    The input can be of type List[Transaction] or List[TxnPostingAccount].
    The account name is extracted from the last posting of each transaction,
    or from TxnPostingAccount.posting.account of each TxnPostingAccount.
    The output is a List[str].
    """

    def transform(self, data: Union[List[TxnPosting], List[Transaction]]):
        return [self._get_posting_account(d) for d in data]

    def _get_posting_account(self, d):
        if isinstance(d, Transaction):
            return d.postings[-1].account
        elif isinstance(d, TxnPostingAccount):
            return d.posting.account


class GetReferencePostingAccount(TransformerMixin, NoFitMixin):
    """
    Scikit-learn transformer to extract the reference account name.  The input
    can be of type List[Transaction] or List[TxnPostingAccount].  The reference
    account name is extracted from the first posting of each transaction.  The
    output is a List[str].
    """

    def transform(self,
                  data: Union[List[TxnPostingAccount], List[Transaction]]):
        return [self._get_posting_account(d) for d in data]

    def _get_posting_account(self, d):
        if isinstance(d, Transaction):
            return d.postings[0].account
        elif isinstance(d, TxnPostingAccount):
            return d.account


class GetDayOfMonth(TransformerMixin, NoFitMixin):
    """
    Scikit-learn transformer to extract the day of month when a transaction
    happened. The input can be of type List[Transaction] or
    List[TxnPostingAccount], the output is a List[Date].
    """

    def transform(self,
                  data: Union[List[TxnPostingAccount], List[Transaction]]):
        return [self._get_day_of_month(d) for d in data]

    def _get_day_of_month(self, d):
        if isinstance(d, Transaction):
            return d.date.day
        elif isinstance(d, TxnPostingAccount):
            return d.txn.date.day
