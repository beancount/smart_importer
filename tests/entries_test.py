import json

from beancount.parser import parser

from smart_importer.entries import (add_suggested_accounts_to_transaction,
                                    add_posting_to_transaction)

TEST_DATA, _, __ = parser.parse_string("""
2016-01-06 * "Farmer Fresh" "Buying groceries"
  Assets:US:BofA:Checking  -10.00 USD
""")
TEST_TRANSACTION = TEST_DATA[0]


def test_add_posting_to_transaction():
    transaction = add_posting_to_transaction(
        TEST_TRANSACTION, "Expenses:Food:Groceries")
    assert transaction.postings[1].account == "Expenses:Food:Groceries"


def test_add_suggested_accounts_to_transaction():
    suggestions = ["Expenses:Food:Groceries",
                   "Expenses:Food:Restaurant",
                   "Expenses:Household",
                   "Expenses:Gifts"]
    transaction = add_suggested_accounts_to_transaction(
        TEST_TRANSACTION, suggestions)
    assert transaction.meta['__suggested_accounts__'] == \
        json.dumps(suggestions)
