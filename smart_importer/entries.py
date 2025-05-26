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
    otherwise it is returned unchanged. Adds empty postings for all the
    accounts - if the account of the single existing posting is found
    in the list of accounts, it is placed there at the first occurence,
    otherwise it is appended at the end.
    """

    if len(transaction.postings) != 1:
        return transaction

    posting = transaction.postings[0]

    new_postings = [
        Posting(account, None, None, None, None, None) for account in accounts
    ]
    if posting.account in accounts:
        new_postings[accounts.index(posting.account)] = posting
    else:
        new_postings.append(posting)

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
