#!/usr/bin/env python3
"""Example import configuration."""

from smart_importer.examples.importer_using_predict_postings.importer import Importer as ImporterUsingPredictPostings

CONFIG = [
    ImporterUsingPredictPostings(['.*']),
]
