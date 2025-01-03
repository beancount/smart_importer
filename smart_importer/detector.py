"""Duplicate detector hook."""

from __future__ import annotations

import logging
from typing import Callable

from beancount.core import data
from beangulp import Importer, similar

from smart_importer.hooks import ImporterHook

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class DuplicateDetector(ImporterHook):
    """Class for duplicate detector importer helpers.

    Args:
        comparator: A functor used to establish the similarity of two entries.
        window_days: The number of days (inclusive) before or after to scan the
        entries to classify against.
    """

    def __init__(
        self,
        comparator: Callable[[data.Directive, data.Directive], bool]
        | None = None,
        window_days: int = 2,
    ) -> None:
        super().__init__()
        self.comparator = comparator
        self.window_days = window_days

    def __call__(
        self,
        importer: Importer,
        file: str,
        imported_entries: data.Directives,
        existing: data.Directives,
    ) -> data.Directives:
        """Add duplicate metadata for imported transactions.

        Args:
            imported_entries: The list of imported entries.
            existing: The list of existing entries as passed to the
                importer.

        Returns:
            A list of entries, modified by this detector.
        """

        duplicate_pairs = similar.find_similar_entries(
            imported_entries,
            existing,
            self.comparator,
            self.window_days,
        )
        # Add a metadata marker to the extracted entries for duplicates.
        duplicate_set = {id(entry) for entry, _ in duplicate_pairs}
        mod_entries = []
        for entry in imported_entries:
            if id(entry) in duplicate_set:
                marked_meta = entry.meta.copy()
                marked_meta["__duplicate__"] = True
                entry = entry._replace(meta=marked_meta)
            mod_entries.append(entry)

        return mod_entries
