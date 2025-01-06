"""Tests for the OldEntryTrimmer"""

from typing import List

import copy
import pytest

from beancount.core.data import Directive, Transaction
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
; 0: old_dup - existing entry before last balance
2016-01-07 * "Farmer Fresh" "Groceries"
  Assets:US:BofA:Checking                                     -10.20 USD
  Expenses:Food:Groceries

; 1: old_non_dup - new entry before last balance
2016-01-08 * "Scammers" "Car warranty extension"
  Assets:US:BofA:Checking                                     -38.36 USD
  Expenses:Scams

; 2: new_dup - existing entry after last balance
2016-01-10 * "Uncle Boons" "Dinner with Mary"
  Assets:US:BofA:Checking                                     -35.00 USD
  Expenses:Food:Restaurant

; 3: new_non_dup - new entry, not seen before
2016-01-11 * "Ye Old Diner" "Lunch with Hasan"
  Assets:US:BofA:Checking                                     -27.30 USD
  Expenses:Food:Restaurant
"""
)


def _marked_duplicate(entry: Transaction):
    entry.meta["__duplicate__"] = True
    return entry


old_dup = _marked_duplicate(copy.deepcopy(example_imported_data[0]))
old_non_dup = copy.deepcopy(example_imported_data[1])
new_dup = _marked_duplicate(copy.deepcopy(example_imported_data[2]))
new_non_dup = copy.deepcopy(example_imported_data[3])


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
    assert importer.extract("foo", existing_entries) == [
        old_dup,
        old_non_dup,
        new_dup,
        new_non_dup,
    ]


def test_trimmer_removes_old_entries_keeps_nondups():
    importer = apply_hooks(
        FakeImporter(example_imported_data),
        [DuplicateDetector(), OldEntryTrimmer(only_trim_duplicates=True)],
    )
    assert importer.extract("foo", existing_entries) == [
        old_non_dup,
        new_dup,
        new_non_dup,
    ]


def test_trimmer_removes_all_old_entries():
    importer = apply_hooks(
        FakeImporter(example_imported_data),
        [DuplicateDetector(), OldEntryTrimmer(only_trim_duplicates=False)],
    )
    assert importer.extract("foo", existing_entries) == [new_dup, new_non_dup]
