"""Machine Learning Helpers."""

import logging
from typing import List, Union, Tuple, NamedTuple
import operator

import numpy
from beancount import loader
from beancount.core.data import Transaction, Posting, filter_txns
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


class NoFitMixin:
    """Mixin that implements a transformer's fit method that returns self."""

    def fit(self, X, y=None):
        return self


class ArrayCaster(BaseEstimator, TransformerMixin, NoFitMixin):
    """
    Helper class for casting data into array shape.
    """

    def transform(self, data):
        return numpy.transpose(numpy.matrix(data))


class Getter(TransformerMixin, NoFitMixin):
    def transform(self,
                  data: Union[List[TxnPostingAccount], List[Transaction]]):
        return [self._getter(d) for d in data]

    def _getter(self, txn):
        if isinstance(txn, TxnPostingAccount):
            txn = txn.txn
        return self._txn_getter(txn)

    def _txn_getter(self, txn):
        pass


class GetPayee(Getter):
    """Payee of the transaction."""

    def _txn_getter(self, txn):
        return txn.payee or ''


class AttrGetter(Getter):
    def __init__(self, attr):
        self._txn_getter = operator.attrgetter(attr)


class GetReferencePostingAccount(Getter):
    """Account of the first posting or of TxnPostingAccount."""

    def _getter(self, txn):
        if isinstance(txn, TxnPostingAccount):
            return txn.account
        return txn.postings[0].account
