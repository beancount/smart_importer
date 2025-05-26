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

    assert _update(["Assets:US:BofA:Checking", "Assets:Other"]) == [
        ("Assets:US:BofA:Checking", True),
        ("Assets:Other", False),
    ]

    assert _update(
        ["Assets:US:BofA:Checking", "Assets:US:BofA:Checking", "Assets:Other"]
    ) == [
        ("Assets:US:BofA:Checking", True),
        ("Assets:US:BofA:Checking", False),
        ("Assets:Other", False),
    ]

    assert _update(["Assets:Other", "Assets:Other2"]) == [
        ("Assets:Other", False),
        ("Assets:Other2", False),
        ("Assets:US:BofA:Checking", True),
    ]

    txn1 = TEST_DATA[1]
    assert isinstance(txn1, Transaction)
    assert update_postings(txn1, ["Assets:Other"]) == txn1
