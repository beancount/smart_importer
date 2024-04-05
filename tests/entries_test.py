# pylint: disable=missing-docstring
import textwrap

from beancount.parser import parser

from smart_importer.entries import update_postings


def test_update_postings_regular():

    test_entries, _, _ = parser.parse_string(
        textwrap.dedent("""
2024-04-04 * "Supermarket ABC" "Groceries"
  Assets:US:BofA:Checking  -100.00 USD
        """))

    transaction = test_entries[0]
    accounts = "Assets:US:BofA:Checking", "Expenses:Food"

    updated_transaction = update_postings(transaction, accounts)

    assert len(updated_transaction.postings) == 2

    # Check if the first posting is unchanged
    assert updated_transaction.postings[0] == transaction.postings[0]

    # Check if the remaining postings are created correctly
    assert updated_transaction.postings[1].account == "Expenses:Food"


def test_update_postings_inverse():

    test_entries, _, _ = parser.parse_string(
        textwrap.dedent("""
2024-04-04 * "Supermarket ABC" "Groceries"
  Assets:US:BofA:Checking  -100.00 USD
        """))

    transaction = test_entries[0]
    accounts = "Expenses:Food", "Assets:US:BofA:Checking"

    updated_transaction = update_postings(transaction, accounts)

    assert len(updated_transaction.postings) == 2

    # Check if the first posting is unchanged
    assert updated_transaction.postings[0] == transaction.postings[0]

    # Check if the remaining postings are created correctly
    assert updated_transaction.postings[1].account == "Expenses:Food"


def test_update_postings_multi():

    test_entries, _, _ = parser.parse_string(
        textwrap.dedent("""
2024-04-04 * "Supermarket ABC" "Groceries"
  Assets:US:BofA:Checking  -100.00 USD
        """))

    transaction = test_entries[0]
    accounts = "Assets:US:BofA:Checking", "Expenses:Food", "Expenses:Clothing"

    updated_transaction = update_postings(transaction, accounts)

    assert len(updated_transaction.postings) == 3

    # Check if the first posting is unchanged
    assert updated_transaction.postings[0] == transaction.postings[0]

    # Check if the remaining postings are created correctly
    assert updated_transaction.postings[1].account == "Expenses:Food"
    assert updated_transaction.postings[2].account == "Expenses:Clothing"
