"""Old entry trimmer hook.

This hook excludes entries in the import that occurred prior to the last
existing balance entry (that is already in the beancount ledger). Since these
imported transactions should already be accounted for, the output of
bean-extract becomes a lot smaller and easier to read. This is especially useful
for financial institutions that do not allow customizing the date range of
exported transactions.

By default, we only trim entries already marked duplicate (e.g., by
`DuplicateDetector`). If you do not use `DuplicateDetector` or want to trim old
entries regardless of the "duplicate" metadata, set `only_trim_duplicates` to
`False`.
"""

from typing import List

import logging
import datetime

from beancount.core.data import Directive, Balance, Transaction


from smart_importer.hooks import ImporterHook


class OldEntryTrimmer(ImporterHook):
    def __init__(self, only_trim_duplicates: bool = True):
        self.only_trim_duplicates = only_trim_duplicates

    def __call__(
        self,
        importer,
        file,
        imported_entries: List[Directive],
        existing_entries: List[Directive],
    ):
        balance_entries: List[Balance] = [
            entry for entry in existing_entries if isinstance(entry, Balance)
        ]
        balance_dates: List[datetime.date] = [
            entry.date for entry in balance_entries
        ]
        last_balance: datetime.date = max(
            balance_dates, default=datetime.date.min
        )

        def should_keep(entry: Directive):
            # Always keep non-transactions.
            if not isinstance(entry, Transaction):
                return True
            # Always keep "new" transactions.
            if entry.date >= last_balance:
                return True
            # Always discard duplicates.
            if entry.meta.get("__duplicate__", False):
                return False
            # At this point, we have an old non-duplicate transaction.
            return self.only_trim_duplicates

        return [entry for entry in imported_entries if should_keep(entry)]
