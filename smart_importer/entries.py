"""Helpers to work with Beancount entry objects."""
import json

from beancount.core.data import Posting
from beancount.core.data import Transaction


def update_postings(transaction, accounts):
    """Update the list of postings of a transaction to match the accounts."""

    if len(transaction.postings) != 1:
        return transaction

    new_postings = [
        Posting(account, None, None, None, None, None) for account in accounts
    ]
    for posting in transaction.postings:
        if posting.account in accounts:
            new_postings[accounts.index(posting.account)] = posting
        else:
            new_postings.append(posting)

    return transaction._replace(postings=new_postings)


def set_entry_attribute(entry, attribute, value, overwrite=False):
    """Set an entry attribute."""
    if value and (not getattr(entry, attribute) or overwrite):
        entry = entry._replace(**{attribute: value})
    return entry


def add_suggestions_to_entry(entry, suggestions, key):
    """Adds a list of suggestions to an entry under entry.meta[key]."""
    entry.meta[key] = json.dumps(suggestions)
    return entry


def merge_non_transaction_entries(imported_entries, enhanced_transactions):
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
