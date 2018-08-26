#!/usr/bin/env python3
"""Example import configuration."""

import logging

from smart_importer.examples.importer_using_predict_postings.importer import Importer as ImporterUsingPredictPostings
from smart_importer.predict_postings import PredictPostings

logger = logging.getLogger(__name__)

CONFIG = [
    ImporterUsingPredictPostings(['.*']),
]

if __name__ == '__main__':
    # configure main log level
    LOG_LEVEL = logging.DEBUG
    logging.basicConfig(level=LOG_LEVEL)

    # configure smart_importer's log level
    smart_importer_logger = logging.getLogger(PredictPostings.__name__)
    smart_importer_logger.setLevel(logging.DEBUG)
