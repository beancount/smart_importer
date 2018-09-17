"""Machine learning pipelines for data extraction."""

from typing import List
import operator

from beancount.core.data import Transaction

import numpy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline


class NoFitMixin:
    """Mixin that implements a transformer's fit method that returns self."""

    def fit(self, *_, **__):
        """A noop."""
        return self


class ArrayCaster(BaseEstimator, TransformerMixin, NoFitMixin):
    """Helper class for casting data into array shape."""

    @staticmethod
    def transform(data):
        """Turn list into numpy array of the necessary shape."""
        return numpy.array(data, ndmin=2).T


class Getter(TransformerMixin, NoFitMixin):
    """Get an entry attribute."""

    def transform(self, data: List[Transaction]):
        """Return list of entry attributes."""
        return [self._getter(d) for d in data]

    def _getter(self, txn):
        raise NotImplementedError


class AttrGetter(Getter):
    """Get a transaction attribute."""
    def __init__(self, attr, default=None):
        self.default = default
        self._txn_getter = operator.attrgetter(attr)

    def _getter(self, txn):
        return self._txn_getter(txn) or self.default


class StringVectorizer(CountVectorizer):
    """Subclass of CountVectorizer that handles empty data."""

    def __init__(self):
        super().__init__(ngram_range=(1, 3))

    def fit_transform(self, raw_documents, y=None):
        try:
            return super().fit_transform(raw_documents, y)
        except ValueError:
            return numpy.zeros(shape=(len(raw_documents), 0))

    def transform(self, raw_documents):
        try:
            return super().transform(raw_documents)
        except ValueError:
            return numpy.zeros(shape=(len(raw_documents), 0))


def get_pipeline(attribute):
    """Make a pipeline for a given entry attribute."""

    if attribute in ['narration', 'payee']:
        return make_pipeline(
            AttrGetter(attribute, ''),
            StringVectorizer(),
        )
    if attribute == 'date.day':
        return make_pipeline(
            AttrGetter('date.day'),
            ArrayCaster(),
        )
    raise ValueError
