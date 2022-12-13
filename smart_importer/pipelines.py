"""Machine learning pipelines for data extraction."""

from __future__ import annotations

import operator

import numpy
from beancount.core.data import Transaction
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline


class NoFitMixin:
    """Mixin that implements a transformer's fit method that returns self."""

    def fit(self, *_, **__):
        """A noop."""
        return self


def txn_attr_getter(attribute_name: str):
    """Return attribute getter for a transaction that also handles metadata."""
    if attribute_name.startswith("meta."):
        meta_attr = attribute_name[5:]

        def getter(txn):
            return txn.meta.get(meta_attr)

        return getter
    return operator.attrgetter(attribute_name)


class NumericTxnAttribute(BaseEstimator, TransformerMixin, NoFitMixin):
    """Get a numeric transaction attribute and vectorize."""

    def __init__(self, attr: str):
        self.attr = attr
        self._txn_getter = txn_attr_getter(attr)

    def transform(self, data: list[Transaction], _y=None):
        """Return list of entry attributes."""
        return numpy.array([self._txn_getter(d) for d in data], ndmin=2).T


class AttrGetter(BaseEstimator, TransformerMixin, NoFitMixin):
    """Get a string transaction attribute."""

    def __init__(self, attr: str, default: str | None = None):
        self.attr = attr
        self.default = default
        self._txn_getter = txn_attr_getter(attr)

    def transform(self, data: list[Transaction], _y=None):
        """Return list of entry attributes."""
        return [self._txn_getter(d) or self.default for d in data]


class StringVectorizer(CountVectorizer):
    """Subclass of CountVectorizer that handles empty data."""

    def __init__(self, tokenizer=None):
        super().__init__(ngram_range=(1, 3), tokenizer=tokenizer)

    def fit_transform(self, raw_documents: list[str], y=None):
        try:
            return super().fit_transform(raw_documents, y)
        except ValueError:
            return numpy.zeros(shape=(len(raw_documents), 0))

    def transform(self, raw_documents: list[str], _y=None):
        try:
            return super().transform(raw_documents)
        except ValueError:
            return numpy.zeros(shape=(len(raw_documents), 0))


def get_pipeline(attribute: str, tokenizer):
    """Make a pipeline for a given entry attribute."""

    if attribute.startswith("date."):
        return NumericTxnAttribute(attribute)

    # Treat all other attributes as strings.
    return make_pipeline(
        AttrGetter(attribute, default=""), StringVectorizer(tokenizer)
    )
