"""Helpers to work with Beancount entry objects."""

from __future__ import annotations

from typing import TYPE_CHECKING

from beancount.core.data import Posting, Transaction

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from beancount.core.data import Directive


def update_postings(
    transaction: Transaction, accounts: list[str]
) -> Transaction:
    """Update the list of postings of a transaction to match the accounts.

    Expects the transaction to be updated to have exactly one posting,
    otherwise it is returned unchanged. The original posting is always
    kept first, and empty postings are appended for all predicted
    accounts (excluding the original posting's account).
    """

    if len(transaction.postings) != 1:
        return transaction

    new_postings = [transaction.postings[0]] + [
        Posting(account, None, None, None, None, None)
        for account in accounts if account != transaction.postings[0].account
    ]

    return transaction._replace(postings=new_postings)


def set_entry_attribute(
    entry: Transaction, attribute: str, value: Any, overwrite: bool = False
) -> Transaction:
    """Set an entry attribute."""
    if value and (not getattr(entry, attribute) or overwrite):
        entry = entry._replace(**{attribute: value})
    return entry


def merge_non_transaction_entries(
    imported_entries: Sequence[Directive],
    enhanced_transactions: Sequence[Directive],
) -> list[Directive]:
    """Merge modified transactions back into a list of entries."""
    enhanced_entries = []
    enhanced_transactions_iter = iter(enhanced_transactions)
    for entry in imported_entries:
        # pylint: disable=isinstance-second-argument-not-valid-type
        if isinstance(entry, Transaction):
            enhanced_entries.append(next(enhanced_transactions_iter))
        else:
            enhanced_entries.append(entry)

    return enhanced_entries
