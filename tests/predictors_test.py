"""Tests for the `PredictPayees` and the `PredictPostings` decorator"""

from __future__ import annotations

from typing import TYPE_CHECKING

from beancount.core.data import Transaction
from beancount.parser import parser
from beangulp.importer import Importer

from smart_importer import PredictPayees, PredictPostings

if TYPE_CHECKING:
    from collections.abc import Sequence

    from beancount.core.data import Directive

TEST_DATA_RAW, _, __ = parser.parse_string(
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

2017-01-14 * "Axe Throwing with Joe"
  Assets:US:BofA:Checking  -13.37 USD
"""
)
TEST_DATA = [t for t in TEST_DATA_RAW if isinstance(t, Transaction)]


TRAINING_DATA, _, __ = parser.parse_string(
    """
2016-01-01 open Assets:US:BofA:Checking USD
2016-01-01 open Expenses:Food:Coffee USD
2016-01-01 open Expenses:Auto:Diesel USD
2016-01-01 open Expenses:Auto:Gas USD
2016-01-01 open Expenses:Food:Groceries USD
2016-01-01 open Expenses:Food:Restaurant USD
2016-01-01 open Expenses:Denylisted USD

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

2016-01-08 * "Axe Throwing with Joe"
  Assets:US:BofA:Checking  -38.36 USD
  Expenses:Denylisted

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
    "Expenses:Food:Groceries",
]

DENYLISTED_ACCOUNTS = ["Expenses:Denylisted"]


class DummyImporter(Importer):
    """A dummy importer for the test cases."""

    def identify(self, filepath: str) -> bool:
        return True

    def account(self, filepath: str) -> str:
        return "Assets:US:BofA:Checking"

    def extract(
        self, filepath: str, existing: list[Directive]
    ) -> list[Directive]:
        return list(TEST_DATA)


def create_dummy_imports(
    data: Sequence[Directive],
) -> list[tuple[str, list[Directive], str, Importer]]:
    """Create the argument list for a beangulp hook."""
    return [("file", list(data), "Assets:US:BofA:Checking", DummyImporter())]


def test_empty_training_data() -> None:
    """
    Verifies that the decorator leaves the narration intact.
    """
    assert (
        PredictPayees().hook(create_dummy_imports(TEST_DATA), [])[0][1]
        == TEST_DATA
    )
    assert (
        PredictPostings().hook(create_dummy_imports(TEST_DATA), [])[0][1]
        == TEST_DATA
    )


def test_no_transactions() -> None:
    """
    Should not crash when passed empty list of transactions.
    """
    PredictPayees().hook([], [])
    PredictPostings().hook([], [])
    PredictPayees().hook([], TRAINING_DATA)
    PredictPostings().hook([], TRAINING_DATA)
    PredictPayees().hook(create_dummy_imports([]), TRAINING_DATA)
    PredictPostings().hook(create_dummy_imports([]), TRAINING_DATA)


def test_unchanged_narrations() -> None:
    """
    Verifies that the decorator leaves the narration intact
    """
    correct_narrations = [transaction.narration for transaction in TEST_DATA]
    extracted_narrations = [
        transaction.narration
        for transaction in PredictPayees().hook(
            create_dummy_imports(TEST_DATA), TRAINING_DATA
        )[0][1]
        if isinstance(transaction, Transaction)
    ]
    assert extracted_narrations == correct_narrations


def test_unchanged_first_posting() -> None:
    """
    Verifies that the decorator leaves the first posting intact
    """
    correct_first_postings = [
        transaction.postings[0] for transaction in TEST_DATA
    ]
    extracted_first_postings = [
        transaction.postings[0]
        for transaction in PredictPayees().hook(
            create_dummy_imports(TEST_DATA), TRAINING_DATA
        )[0][1]
        if isinstance(transaction, Transaction)
    ]
    assert extracted_first_postings == correct_first_postings


def test_payee_predictions() -> None:
    """
    Verifies that the decorator adds predicted postings.
    """
    transactions = PredictPayees().hook(
        create_dummy_imports(TEST_DATA), TRAINING_DATA
    )[0][1]
    predicted_payees = [
        transaction.payee
        for transaction in transactions
        if isinstance(transaction, Transaction)
    ]
    assert predicted_payees == PAYEE_PREDICTIONS


def test_account_predictions() -> None:
    """
    Verifies that the decorator adds predicted postings.
    """
    predicted_accounts = [
        entry.postings[-1].account
        for entry in PredictPostings(
            denylist_accounts=DENYLISTED_ACCOUNTS
        ).hook(create_dummy_imports(TEST_DATA), TRAINING_DATA)[0][1]
        if isinstance(entry, Transaction)
    ]
    assert predicted_accounts == ACCOUNT_PREDICTIONS


def test_account_predictions_wrap() -> None:
    """
    Verifies account prediction using the wrap method instead of the beangulp hook
    """
    wrapped_importer = PredictPostings(
        denylist_accounts=DENYLISTED_ACCOUNTS
    ).wrap(DummyImporter())
    entries = wrapped_importer.extract("dummyFile", TRAINING_DATA)
    print(entries)
    predicted_accounts = [
        entry.postings[-1].account
        for entry in entries
        if isinstance(entry, Transaction)
    ]
    assert predicted_accounts == ACCOUNT_PREDICTIONS


def test_account_predictions_multiple() -> None:
    """
    Verifies that it's possible to predict multiple importer results
    """
    predicted_results = PredictPostings(
        denylist_accounts=DENYLISTED_ACCOUNTS
    ).hook(
        [
            (
                "file1",
                list(TEST_DATA),
                "Assets:US:BofA:Checking",
                DummyImporter(),
            ),
            (
                "file1",
                list(TEST_DATA),
                "Assets:US:BofA:Checking",
                DummyImporter(),
            ),
        ],
        TRAINING_DATA,
    )

    assert len(predicted_results) == 2
    predicted_accounts1 = [
        entry.postings[-1].account
        for entry in predicted_results[0][1]
        if isinstance(entry, Transaction)
    ]
    predicted_accounts2 = [
        entry.postings[-1].account
        for entry in predicted_results[1][1]
        if isinstance(entry, Transaction)
    ]
    assert predicted_accounts1 == ACCOUNT_PREDICTIONS
    assert predicted_accounts2 == ACCOUNT_PREDICTIONS
