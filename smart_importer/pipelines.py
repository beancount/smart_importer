"""Machine learning pipelines for data extraction."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import numpy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from beancount.core.data import Transaction


class NoFitMixin:
    """Mixin that implements a transformer's fit method that returns self."""

    def fit(self, *_: Any, **__: Any) -> Any:
        """A noop."""
        return self


def txn_attr_getter(attribute_name: str) -> Callable[[Transaction], Any]:
    """Return attribute getter for a transaction that also handles metadata."""
    if attribute_name.startswith("meta."):
        meta_attr = attribute_name[5:]

        def getter(txn: Transaction) -> Any:
            return txn.meta.get(meta_attr)

        return getter
    return operator.attrgetter(attribute_name)


class NumericTxnAttribute(BaseEstimator, TransformerMixin, NoFitMixin):  # type: ignore[misc]
    """Get a numeric transaction attribute and vectorize."""

    def __init__(self, attr: str) -> None:
        self.attr = attr
        self._txn_getter = txn_attr_getter(attr)

    def transform(
        self, data: list[Transaction], _y: None = None
    ) -> numpy.ndarray[tuple[int, ...], Any]:
        """Return list of entry attributes."""
        return numpy.array([self._txn_getter(d) for d in data], ndmin=2).T


class AttrGetter(BaseEstimator, TransformerMixin, NoFitMixin):  # type: ignore[misc]
    """Get a string transaction attribute."""

    def __init__(self, attr: str, default: str | None = None) -> None:
        self.attr = attr
        self.default = default
        self._txn_getter = txn_attr_getter(attr)

    def transform(self, data: list[Transaction], _y: None = None) -> list[Any]:
        """Return list of entry attributes."""
        return [self._txn_getter(d) or self.default for d in data]


class StringVectorizer(CountVectorizer):  # type: ignore[misc]
    """Subclass of CountVectorizer that handles empty data."""

    def __init__(
        self, tokenizer: Callable[[str], list[str]] | None = None
    ) -> None:
        super().__init__(ngram_range=(1, 3), tokenizer=tokenizer)

    def fit_transform(self, raw_documents: list[str], y: None = None) -> Any:
        try:
            return super().fit_transform(raw_documents, y)
        except ValueError:
            return numpy.zeros(shape=(len(raw_documents), 0))

    def transform(self, raw_documents: list[str], _y: None = None) -> Any:
        try:
            return super().transform(raw_documents)
        except ValueError:
            return numpy.zeros(shape=(len(raw_documents), 0))


def get_pipeline(
    attribute: str, tokenizer: Callable[[str], list[str]] | None
) -> Any:
    """Make a pipeline for a given entry attribute."""

    if attribute.startswith("date."):
        return NumericTxnAttribute(attribute)

    # Treat all other attributes as strings.
    return make_pipeline(
        AttrGetter(attribute, default=""), StringVectorizer(tokenizer)
    )
