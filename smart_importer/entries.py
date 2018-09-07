"""Helpers to work with Beancount entry objects."""

import json
from typing import List

from beancount.core.data import Transaction, Posting


def add_posting_to_transaction(transaction: Transaction,
                               account: str) -> Transaction:
    """Adds an empty posting with the given account to a transaction."""

    if len(transaction.postings) != 1:
        return transaction

    additional_posting: Posting
    additional_posting = Posting(account, None, None, None, None, None)
    transaction.postings.append(additional_posting)
    return transaction


def add_payee_to_transaction(transaction: Transaction,
                             payee: str,
                             overwrite=False) -> Transaction:
    """Sets a transactions's payee."""
    if not transaction.payee or overwrite:
        transaction = transaction._replace(payee=payee)
    return transaction


METADATA_KEY_SUGGESTED_ACCOUNTS = '__suggested_accounts__'
METADATA_KEY_SUGGESTED_PAYEES = '__suggested_payees__'


def add_suggested_accounts_to_transaction(
        transaction: Transaction, suggestions: List[str]) -> Transaction:
    """Adds suggested related accounts to a transaction."""
    return _add_suggestions_to_transaction(
        transaction, suggestions, key=METADATA_KEY_SUGGESTED_ACCOUNTS)


def add_suggested_payees_to_transaction(transaction: Transaction,
                                        suggestions: List[str]) -> Transaction:
    """Adds suggested payees to a transaction."""
    return _add_suggestions_to_transaction(
        transaction, suggestions, key=METADATA_KEY_SUGGESTED_PAYEES)


def _add_suggestions_to_transaction(transaction: Transaction,
                                    suggestions: List[str],
                                    key='__suggestions__'):
    """
    Adds a list of suggestions to a transaction under transaction.meta[key].
    """
    meta = transaction.meta
    meta[key] = json.dumps(suggestions)
    transaction = transaction._replace(meta=meta)
    return transaction


def merge_non_transaction_entries(imported_entries, enhanced_transactions):
    enhanced_entries = []
    enhanced_transactions_iter = iter(enhanced_transactions)
    for entry in imported_entries:
        if isinstance(entry, Transaction):
            enhanced_entries.append(next(enhanced_transactions_iter))
        else:
            enhanced_entries.append(entry)

    return enhanced_entries
