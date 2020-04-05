"""Tests for the `PredictPostings` decorator"""
# pylint: disable=missing-docstring
import os
import pprint
import re

import pytest
from beancount.core.compare import stable_hash_namedtuple
from beancount.ingest.importer import ImporterProtocol
from beancount.parser import parser

from smart_importer import apply_hooks
from smart_importer import PredictPostings


def _hash(entry):
    return stable_hash_namedtuple(entry, ignore={"meta", "units"})


def _load_testset(testset):
    path = os.path.join(
        os.path.dirname(__file__), "data", testset + ".beancount"
    )
    with open(path, "r") as test_file:
        _, *sections = re.split(r"# [A-Z]+\n", test_file.read())
    parsed_sections = []
    for section in sections:
        entries, errors, __ = parser.parse_string(section)
        assert not errors
        parsed_sections.append(entries)
    assert len(parsed_sections) == 3
    return parsed_sections


@pytest.mark.parametrize(
    "testset", ["simple", "single-account", "multiaccounts"]
)
def test_testset(testset):
    # pylint: disable=unbalanced-tuple-unpacking
    imported, training_data, expected = _load_testset(testset)

    class DummyImporter(ImporterProtocol):
        def extract(self, file, existing_entries=None):
            return imported

    importer = DummyImporter()
    apply_hooks(importer, [PredictPostings()])
    imported_transactions = importer.extract(
        "dummy-data", existing_entries=training_data
    )

    for txn1, txn2 in zip(imported_transactions, expected):
        if _hash(txn1) != _hash(txn2):
            pprint.pprint(txn1)
            pprint.pprint(txn2)
            assert False
