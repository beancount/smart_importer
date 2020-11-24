"""Tests for the Machine Learning Helpers."""
# pylint: disable=missing-docstring
from beancount.parser import parser

from smart_importer.pipelines import AttrGetter

TEST_DATA, _, __ = parser.parse_string(
    """
2016-01-01 open Assets:US:BofA:Checking USD
2016-01-01 open Expenses:Food:Groceries USD
2016-01-01 open Expenses:Food:Coffee USD

2016-01-06 * "Farmer Fresh" "Buying groceries"
  Assets:US:BofA:Checking  -10.00 USD

2016-01-07 * "Starbucks" "Coffee"
  Assets:US:BofA:Checking  -4.00 USD
  Expenses:Food:Coffee

2016-01-07 * "Farmer Fresh" "Groceries"
  Assets:US:BofA:Checking  -11.20 USD
  Expenses:Food:Groceries

2016-01-08 * "Gimme Coffee" "Coffee"
  Assets:US:BofA:Checking  -3.50 USD
  Expenses:Food:Coffee
"""
)
TEST_TRANSACTIONS = TEST_DATA[3:]
TEST_TRANSACTION = TEST_TRANSACTIONS[0]


def test_get_payee():
    assert AttrGetter("payee").transform(TEST_TRANSACTIONS) == [
        "Farmer Fresh",
        "Starbucks",
        "Farmer Fresh",
        "Gimme Coffee",
    ]


def test_get_narration():
    assert AttrGetter("narration").transform(TEST_TRANSACTIONS) == [
        "Buying groceries",
        "Coffee",
        "Groceries",
        "Coffee",
    ]


def test_get_metadata():
    txn = TEST_TRANSACTION
    txn.meta["attr"] = "value"
    assert AttrGetter("meta.attr").transform([txn]) == ["value"]
    assert AttrGetter("meta.attr", "default").transform(TEST_TRANSACTIONS) == [
        "value",
        "default",
        "default",
        "default",
    ]


def test_get_day_of_month():
    assert AttrGetter("date.day").transform(TEST_TRANSACTIONS) == [6, 7, 7, 8]
