"""Tests for the `PredictPayees` and the `PredictPostings` decorator"""
# pylint: disable=missing-docstring
from beancount.ingest.importer import ImporterProtocol
from beancount.parser import parser

from smart_importer import PredictPayees
from smart_importer import PredictPostings
from smart_importer.hooks import apply_hooks

TEST_DATA, _, __ = parser.parse_string(
    """
2017-01-06 * "Farmer Fresh" "Buying groceries"
  Assets:US:BofA:Checking  -2.50 USD

2017-01-07 * "Groceries"
  Assets:US:BofA:Checking  -10.20 USD

2017-01-10 * "" "Eating out with Joe"
  Assets:US:BofA:Checking  -38.36 USD

2017-01-10 * "Dinner with Martin"
  Assets:US:BofA:Checking  -35.00 USD

2017-01-10 * "Groceries"
  Assets:US:BofA:Checking  -53.70 USD

2017-01-10 * "Gimme Coffee" "Coffee"
  Assets:US:BofA:Checking  -5.00 USD

2017-01-12 * "Uncle Boons" ""
  Assets:US:BofA:Checking  -27.00 USD

2017-01-13 * "Gas Quick"
  Assets:US:BofA:Checking  -17.45 USD
"""
)

TRAINING_DATA, _, __ = parser.parse_string(
    """
2016-01-01 open Assets:US:BofA:Checking USD
2016-01-01 open Expenses:Food:Coffee USD
2016-01-01 open Expenses:Auto:Diesel USD
2016-01-01 open Expenses:Auto:Gas USD
2016-01-01 open Expenses:Food:Groceries USD
2016-01-01 open Expenses:Food:Restaurant USD

2016-01-06 * "Farmer Fresh" "Buying groceries"
  Assets:US:BofA:Checking  -2.50 USD
  Expenses:Food:Groceries

2016-01-07 * "Starbucks" "Coffee"
  Assets:US:BofA:Checking  -4.00 USD
  Expenses:Food:Coffee

2016-01-07 * "Farmer Fresh" "Groceries"
  Assets:US:BofA:Checking  -10.20 USD
  Expenses:Food:Groceries

2016-01-07 * "Gimme Coffee" "Coffee"
  Assets:US:BofA:Checking  -3.50 USD
  Expenses:Food:Coffee

2016-01-07 * "Gas Quick"
  Assets:US:BofA:Checking  -22.79 USD
  Expenses:Auto:Diesel

2016-01-08 * "Uncle Boons" "Eating out with Joe"
  Assets:US:BofA:Checking  -38.36 USD
  Expenses:Food:Restaurant

2016-01-10 * "Walmarts" "Groceries"
  Assets:US:BofA:Checking  -53.70 USD
  Expenses:Food:Groceries

2016-01-10 * "Gimme Coffee" "Coffee"
  Assets:US:BofA:Checking  -6.19 USD
  Expenses:Food:Coffee

2016-01-10 * "Gas Quick"
  Assets:US:BofA:Checking  -21.60 USD
  Expenses:Auto:Diesel

2016-01-10 * "Uncle Boons" "Dinner with Mary"
  Assets:US:BofA:Checking  -35.00 USD
  Expenses:Food:Restaurant

2016-01-11 close Expenses:Auto:Diesel

2016-01-11 * "Farmer Fresh" "Groceries"
  Assets:US:BofA:Checking  -30.50 USD
  Expenses:Food:Groceries

2016-01-12 * "Gas Quick"
  Assets:US:BofA:Checking  -24.09 USD
  Expenses:Auto:Gas
"""
)

PAYEE_PREDICTIONS = [
    "Farmer Fresh",
    "Farmer Fresh",
    "Uncle Boons",
    "Uncle Boons",
    "Farmer Fresh",
    "Gimme Coffee",
    "Uncle Boons",
    None,
]

ACCOUNT_PREDICTIONS = [
    "Expenses:Food:Groceries",
    "Expenses:Food:Groceries",
    "Expenses:Food:Restaurant",
    "Expenses:Food:Restaurant",
    "Expenses:Food:Groceries",
    "Expenses:Food:Coffee",
    "Expenses:Food:Groceries",
    "Expenses:Auto:Gas",
]


class BasicTestImporter(ImporterProtocol):
    def extract(self, file, existing_entries=None):
        if file == "dummy-data":
            return TEST_DATA
        if file == "empty":
            return []
        assert False
        return []

    def file_account(self, file):
        return "Assets:US:BofA:Checking"


PAYEE_IMPORTER = apply_hooks(
    BasicTestImporter(), [PredictPayees(suggest=True)]
)
POSTING_IMPORTER = apply_hooks(
    BasicTestImporter(), [PredictPostings(suggest=True)]
)


def test_empty_training_data():
    """
    Verifies that the decorator leaves the narration intact.
    """
    assert POSTING_IMPORTER.extract("dummy-data") == TEST_DATA
    assert PAYEE_IMPORTER.extract("dummy-data") == TEST_DATA


def test_no_transactions():
    """
    Should not crash when passed empty list of transactions.
    """
    POSTING_IMPORTER.extract("empty")
    PAYEE_IMPORTER.extract("empty")
    POSTING_IMPORTER.extract("empty", existing_entries=TRAINING_DATA)
    PAYEE_IMPORTER.extract("empty", existing_entries=TRAINING_DATA)


def test_unchanged_narrations():
    """
    Verifies that the decorator leaves the narration intact
    """
    correct_narrations = [transaction.narration for transaction in TEST_DATA]
    extracted_narrations = [
        transaction.narration
        for transaction in PAYEE_IMPORTER.extract(
            "dummy-data", existing_entries=TRAINING_DATA
        )
    ]
    assert extracted_narrations == correct_narrations


def test_unchanged_first_posting():
    """
    Verifies that the decorator leaves the first posting intact
    """
    correct_first_postings = [
        transaction.postings[0] for transaction in TEST_DATA
    ]
    extracted_first_postings = [
        transaction.postings[0]
        for transaction in PAYEE_IMPORTER.extract(
            "dummy-data", existing_entries=TRAINING_DATA
        )
    ]
    assert extracted_first_postings == correct_first_postings


def test_payee_predictions():
    """
    Verifies that the decorator adds predicted postings.
    """
    transactions = PAYEE_IMPORTER.extract(
        "dummy-data", existing_entries=TRAINING_DATA
    )
    predicted_payees = [transaction.payee for transaction in transactions]
    assert predicted_payees == PAYEE_PREDICTIONS


def test_payee_suggestions():
    """
    Verifies that the decorator adds suggestions about accounts
    """
    transactions = PAYEE_IMPORTER.extract(
        "dummy-data", existing_entries=TRAINING_DATA
    )
    for transaction in transactions:
        assert transaction.meta["__suggested_payees__"]


def test_account_predictions():
    """
    Verifies that the decorator adds predicted postings.
    """
    predicted_accounts = [
        entry.postings[-1].account
        for entry in POSTING_IMPORTER.extract(
            "dummy-data", existing_entries=TRAINING_DATA
        )
    ]
    assert predicted_accounts == ACCOUNT_PREDICTIONS


def test_account_suggestions():
    """
    Verifies that the decorator adds suggestions.
    """
    for transaction in POSTING_IMPORTER.extract(
        "dummy-data", existing_entries=TRAINING_DATA
    ):
        assert transaction.meta["__suggested_accounts__"]
