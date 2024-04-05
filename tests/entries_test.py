"""Tests for the entry helpers."""

# pylint: disable=missing-docstring

from __future__ import annotations

from beancount.core.data import Transaction
from beancount.parser import parser

from smart_importer.entries import update_postings

TEST_DATA, _errors, _options = parser.parse_string(
    """
2016-01-06 * "Farmer Fresh" "Buying groceries"
  Assets:US:BofA:Checking  -10.00 USD

2016-01-06 * "Farmer Fresh" "Buying groceries"
  Assets:US:BofA:Checking  -10.00 USD
  Assets:US:BofA:Checking   10.00 USD
"""
)


def test_update_postings() -> None:
    txn0 = TEST_DATA[0]
    assert isinstance(txn0, Transaction)

    def _update(accounts: list[str]) -> list[tuple[str, bool]]:
        """Update, get accounts and whether this is the original posting."""
        updated = update_postings(txn0, accounts)
        return [(p.account, p is txn0.postings[0]) for p in updated.postings]

    # Original posting is always kept first, predicted accounts appended.
    assert _update(["Assets:US:BofA:Checking", "Assets:Other"]) == [
        ("Assets:US:BofA:Checking", True),
        ("Assets:Other", False),
    ]

    # Duplicate predicted account: original still first, one predicted copy.
    assert _update(
        ["Assets:US:BofA:Checking", "Assets:US:BofA:Checking", "Assets:Other"]
    ) == [
        ("Assets:US:BofA:Checking", True),
        ("Assets:Other", False),
    ]

    # Original account not in prediction list: original first, rest appended.
    assert _update(["Assets:Other", "Assets:Other2"]) == [
        ("Assets:US:BofA:Checking", True),
        ("Assets:Other", False),
        ("Assets:Other2", False),
    ]

    # Multi-posting transactions are left unchanged.
    txn1 = TEST_DATA[1]
    assert isinstance(txn1, Transaction)
    assert update_postings(txn1, ["Assets:Other"]) == txn1


def test_update_postings_inverse() -> None:
    """Predicted account order doesn't affect result — original stays first."""
    txn0 = TEST_DATA[0]
    assert isinstance(txn0, Transaction)

    updated = update_postings(txn0, ["Expenses:Food", "Assets:US:BofA:Checking"])

    assert len(updated.postings) == 2
    assert updated.postings[0] is txn0.postings[0]
    assert updated.postings[1].account == "Expenses:Food"


def test_update_postings_multi_predicted() -> None:
    """Multiple predicted counter-accounts are appended after original."""
    txn0 = TEST_DATA[0]
    assert isinstance(txn0, Transaction)

    updated = update_postings(
        txn0, ["Assets:US:BofA:Checking", "Expenses:Food", "Expenses:Clothing"]
    )

    assert len(updated.postings) == 3
    assert updated.postings[0] is txn0.postings[0]
    assert updated.postings[1].account == "Expenses:Food"
    assert updated.postings[2].account == "Expenses:Clothing"
