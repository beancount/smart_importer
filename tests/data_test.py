"""Tests for the `PredictPostings` decorator"""
# pylint: disable=missing-docstring
import os
import pprint
import re

import jieba
import pytest
from beancount.core.compare import stable_hash_namedtuple
from beancount.ingest.importer import ImporterProtocol
from beancount.parser import parser

from smart_importer import PredictPostings, apply_hooks

jieba.initialize()


def chinese_string_tokenizer(pre_tokenizer_string):
    return list(jieba.cut(pre_tokenizer_string))


def _hash(entry):
    return stable_hash_namedtuple(entry, ignore={"meta", "units"})


def _load_testset(testset):
    path = os.path.join(
        os.path.dirname(__file__), "data", testset + ".beancount"
    )
    with open(path, encoding="utf-8") as test_file:
        _, *sections = re.split(r"# [A-Z]+\n", test_file.read())
    parsed_sections = []
    for section in sections:
        entries, errors, __ = parser.parse_string(section)
        assert not errors
        parsed_sections.append(entries)
    assert len(parsed_sections) == 3
    return parsed_sections


@pytest.mark.parametrize(
    "testset, string_tokenizer",
    [
        ("simple", None),
        ("single-account", None),
        ("multiaccounts", None),
        ("chinese", chinese_string_tokenizer),
    ],
)
def test_testset(testset, string_tokenizer):
    # pylint: disable=unbalanced-tuple-unpacking
    imported, training_data, expected = _load_testset(testset)

    class DummyImporter(ImporterProtocol):
        def extract(self, file, existing_entries=None):
            return imported

    importer = DummyImporter()
    apply_hooks(importer, [PredictPostings(string_tokenizer=string_tokenizer)])
    imported_transactions = importer.extract(
        "dummy-data", existing_entries=training_data
    )

    for txn1, txn2 in zip(imported_transactions, expected):
        if _hash(txn1) != _hash(txn2):
            pprint.pprint(txn1)
            pprint.pprint(txn2)
            assert False
