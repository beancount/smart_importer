"""Tests for the `PredictPostings` decorator"""

from __future__ import annotations

# pylint: disable=missing-docstring
import os
import pprint
import re
from typing import Callable

import pytest
from beancount.core import data
from beancount.core.compare import stable_hash_namedtuple
from beancount.parser import parser

from smart_importer import PredictPostings


def chinese_string_tokenizer(pre_tokenizer_string: str) -> list[str]:
    jieba = pytest.importorskip("jieba")
    jieba.initialize()
    return list(jieba.cut(pre_tokenizer_string))


def _hash(entry: data.Directive) -> str:
    return stable_hash_namedtuple(entry, ignore={"meta", "units"})


def _load_testset(
    testset: str,
) -> tuple[data.Directives, data.Directives, data.Directives]:
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
    return tuple(parsed_sections)


@pytest.mark.parametrize(
    "testset, account, string_tokenizer",
    [
        ("simple", "Assets:US:BofA:Checking", None),
        ("single-account", "Assets:US:BofA:Checking", None),
        ("multiaccounts", "Assets:US:EUR", None),
        ("chinese", "Assets:US:BofA:Checking", chinese_string_tokenizer),
    ],
)
def test_testset(
    testset: str, account: str, string_tokenizer: Callable[[str], list[str]]
) -> None:
    # pylint: disable=unbalanced-tuple-unpacking
    imported, training_data, expected = _load_testset(testset)

    imported_transactions = PredictPostings(
        string_tokenizer=string_tokenizer
    ).hook([("file", imported, account, "importer")], training_data)

    for txn1, txn2 in zip(imported_transactions[0][1], expected):
        if _hash(txn1) != _hash(txn2):
            pprint.pprint(txn1)
            pprint.pprint(txn2)
            assert False
