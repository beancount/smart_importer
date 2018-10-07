"""Duplicate detector importer decorators."""

import logging

from beancount.ingest import similar

from smart_importer.decorator import ImporterDecorator

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class DuplicateDetector(ImporterDecorator):
    """Class for duplicate detector importer helpers.
    """

    def __init__(self):
        super().__init__()

    def main(self, imported_entries, existing_entries):
        """Add duplicate metadata for imported transactions.

        Args:
            imported_entries: The list of imported entries.
            existing_entries: The list of existing entries as passed to the
                importer.

        Returns:
            A list of entries, modified by this detector.
        """

        duplicate_pairs = similar.find_similar_entries(imported_entries, existing_entries)
        # Add a metadata marker to the extracted entries for duplicates.
        duplicate_set = set(id(entry) for entry, _ in duplicate_pairs)
        mod_entries = []
        for entry in imported_entries:
            if id(entry) in duplicate_set:
                marked_meta = entry.meta.copy()
                marked_meta['__duplicate__'] = True
                entry = entry._replace(meta=marked_meta)
            mod_entries.append(entry)

        return mod_entries
