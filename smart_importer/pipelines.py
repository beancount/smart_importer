"""Machine learning pipelines for data extraction."""

from typing import List, Union, NamedTuple
import operator

from beancount.core.data import Transaction, Posting

import numpy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline


TxnPostingAccount = NamedTuple('TxnPostingAccount',
                               [('txn', Transaction), ('posting', Posting),
                                ('account', str)])


class NoFitMixin:
    """Mixin that implements a transformer's fit method that returns self."""

    def fit(self, *_, **__):
        return self


class ArrayCaster(BaseEstimator, TransformerMixin, NoFitMixin):
    """
    Helper class for casting data into array shape.
    """

    @staticmethod
    def transform(data):
        return numpy.transpose(numpy.matrix(data))


class Getter(TransformerMixin, NoFitMixin):
    def transform(self,
                  data: Union[List[TxnPostingAccount], List[Transaction]]):
        return [self._getter(d) for d in data]

    def _getter(self, txn):
        raise NotImplementedError


class AttrGetter(Getter):
    """Get a transaction attribute."""
    def __init__(self, attr, default=None):
        self.default = default
        self._txn_getter = operator.attrgetter(attr)

    def _getter(self, txn):
        if isinstance(txn, TxnPostingAccount):
            txn = txn.txn
        return self._txn_getter(txn) or self.default


class GetReferencePostingAccount(Getter):
    """Account of the first posting or of TxnPostingAccount."""

    def _getter(self, txn):
        if isinstance(txn, TxnPostingAccount):
            return txn.account
        return txn.postings[0].account


PIPELINES = {
    'narration': make_pipeline(
        AttrGetter('narration', ''),
        CountVectorizer(ngram_range=(1, 3))
    ),
    'payee': make_pipeline(
        AttrGetter('payee', ''),
        CountVectorizer(ngram_range=(1, 3))
    ),
    'first_posting_account': make_pipeline(
        GetReferencePostingAccount(),
        CountVectorizer(ngram_range=(1, 3))
    ),
    'date.day': make_pipeline(
        AttrGetter('date.day'),
        ArrayCaster(),
    ),
}
