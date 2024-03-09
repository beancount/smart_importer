"""Tests for the OldEntryTrimmer"""

from typing import List

import copy
import pytest

from beancount.core.data import Directive
from beancount.ingest.importer import ImporterProtocol
from beancount.parser import parser

from smart_importer import apply_hooks
from smart_importer.trimmer import OldEntryTrimmer
from smart_importer.detector import DuplicateDetector


existing_entries, _, _ = parser.parse_string(
    """
2016-01-01 open Assets:US:BofA:Checking USD
2016-01-01 open Equity:Initial-Balance:US:BofA:Checking USD
2016-01-01 open Expenses:Food:Groceries USD
2016-01-01 open Expenses:Food:Restaurant USD
2016-01-01 open Expenses:Scams USD

2016-01-01 * "Initial Balance"
  Assets:US:BofA:Checking                                        100 USD
  Equity:Initial-Balance:US:BofA:Checking

2016-01-02 balance Assets:US:BofA:Checking                       100 USD

2016-01-06 * "Farmer Fresh" "Buying groceries"
  Assets:US:BofA:Checking                                      -2.50 USD
  Expenses:Food:Groceries

2016-01-07 balance Assets:US:BofA:Checking                     97.50 USD

2016-01-07 * "Farmer Fresh" "Groceries"
  Assets:US:BofA:Checking                                     -10.20 USD
  Expenses:Food:Groceries

2016-01-08 balance Assets:US:BofA:Checking                     87.30 USD

2016-01-08 * "Uncle Boons" "Eating out with Joe"
  Assets:US:BofA:Checking                                     -38.36 USD
  Expenses:Food:Restaurant

2016-01-09 balance Assets:US:BofA:Checking                     48.94 USD

2016-01-10 * "Uncle Boons" "Dinner with Mary"
  Assets:US:BofA:Checking                                     -35.00 USD
  Expenses:Food:Restaurant

"""
)

example_imported_data, _, _ = parser.parse_string(
    """
; 0: existing entry before last balance
2016-01-07 * "Farmer Fresh" "Groceries"
  Assets:US:BofA:Checking                                     -10.20 USD
  Expenses:Food:Groceries

; 1: new entry before last balance
2016-01-08 * "Scammers" "Car warranty extension"
  Assets:US:BofA:Checking                                     -38.36 USD
  Expenses:Scams

; 2: existing entry after last balance
2016-01-10 * "Uncle Boons" "Dinner with Mary"
  Assets:US:BofA:Checking                                     -35.00 USD
  Expenses:Food:Restaurant

; 3: new entry
2016-01-11 * "Ye Old Diner" "Lunch with Hasan"
  Assets:US:BofA:Checking                                     -27.30 USD
  Expenses:Food:Restaurant
"""
)


def example_data_subset(
    *indices: List[int], duplicates: None
) -> List[Directive]:
    duplicates = set(duplicates or [])
    output = []
    for i in indices:
        entry = copy.deepcopy(example_imported_data[i])
        if i in duplicates:
            entry.meta["__duplicate__"] = True
        output.append(entry)
    return output


class FakeImporter(ImporterProtocol):
    def __init__(self, postings: List[Directive]):
        self.postings = postings

    def extract(self, file, existing_entries=None):
        return self.postings

    def file_account(self, file):
        return "Assets:US:BofA:Checking"


def test_importer_returns_all_data():
    importer = apply_hooks(
        FakeImporter(example_imported_data),
        [DuplicateDetector()],
    )
    assert importer.extract("foo", existing_entries) == example_data_subset(
        0, 1, 2, 3, duplicates=[0, 2]
    )


def test_trimmer_removes_old_entries_keeps_nondups():
    importer = apply_hooks(
        FakeImporter(example_imported_data),
        [DuplicateDetector(), OldEntryTrimmer(only_trim_duplicates=True)],
    )
    # Here, we *do* keep 1 because it's not a duplicate, even though it's old.
    assert importer.extract("foo", existing_entries) == example_data_subset(
        1, 2, 3, duplicates=[0, 2]
    )


def test_trimmer_removes_all_old_entries():
    importer = apply_hooks(
        FakeImporter(example_imported_data),
        [DuplicateDetector(), OldEntryTrimmer(only_trim_duplicates=False)],
    )
    # Here, we discard 1 even though it's a duplicate.
    assert importer.extract("foo", existing_entries) == example_data_subset(
        2, 3, duplicates=[0, 2]
    )
