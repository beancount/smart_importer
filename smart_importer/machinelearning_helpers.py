"""Machine Learning Helpers."""

import logging
from typing import List, Union, Tuple

from beancount import loader
from beancount.core.data import Transaction, filter_txns

logger = logging.getLogger(__name__)


def load_training_data(
        training_data: Union[List[Transaction], str],
        known_account: str = None,
        existing_entries: List[Tuple] = None) -> List[Transaction]:
    """Load training data.

    :param training_data: The training data that shall be loaded.
        Can be provided as a string (the filename pointing to a beancount
            file),
        or a list of Beancount entries
    :param known_account: Optional filter for the training data.
        If provided, the training data is filtered to only include transactions
        that involve the specified account.
    :param existing_entries: Optional existing entries to use instead of
        explicit training_data
    :return: A list of Beancount entries.
    """
    if not training_data and existing_entries:
        logger.debug("Using existing entries for training data")
        training_data = existing_entries
    elif isinstance(training_data, str):
        logger.debug(f"Reading training data from file \"{training_data}\".")
        training_data, _, __ = loader.load_file(training_data)
    logger.debug("Finished reading training data.")
    if training_data:
        training_data = list(filter_txns(training_data))
    if known_account:
        training_data = [
            txn for txn in training_data
            if any([pos.account == known_account for pos in txn.postings])
        ]
        logger.debug(
            f"After filtering for account {known_account}, "
            f"the training data consists of {len(training_data)} entries.")
    return training_data
