"""Machine learning importer decorators."""
# pylint: disable=unsubscriptable-object

import logging
import threading
from typing import Dict
from typing import List
from typing import Union

from beancount.core.data import ALL_DIRECTIVES
from beancount.core.data import Transaction, Open, Close
from beancount.core.data import filter_txns
from beancount.core.data import sorted as beancount_sorted
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from smart_importer.entries import merge_non_transaction_entries
from smart_importer.entries import set_entry_attribute
from smart_importer.hooks import ImporterHook
from smart_importer.pipelines import get_pipeline

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class EntryPredictor(ImporterHook):
    """Base class for machine learning importer helpers.

    Args:
        predict: Whether to add predictions to the entries.
        overwrite: When an attribute is predicted but already exists on an
            entry, overwrite the existing one.
    """

    # pylint: disable=too-many-instance-attributes

    weights: Dict[str, int] = {}
    attribute = None

    def __init__(self, predict=True, overwrite=False):
        super().__init__()
        self.training_data = None
        self.open_accounts = {}
        self.pipeline = None
        self.is_fitted = False
        self.lock = threading.Lock()
        self.account = None

        self.predict = predict
        self.overwrite = overwrite

    def __call__(self, importer, file, imported_entries, existing_entries):
        """Predict attributes for imported transactions.

        Args:
            imported_entries: The list of imported entries.
            existing_entries: The list of existing entries as passed to the
                importer - will be used as training data.

        Returns:
            A list of entries, modified by this predictor.
        """

        self.account = importer.file_account(file)
        self.load_training_data(existing_entries)
        with self.lock:
            self.define_pipeline()
            self.train_pipeline()
            return self.process_entries(imported_entries)

    def load_open_accounts(self, existing_entries):
        """Return map of accounts which have been opened but not closed."""
        account_map = {}
        if not existing_entries:
            return

        for entry in beancount_sorted(existing_entries):
            # pylint: disable=isinstance-second-argument-not-valid-type
            if isinstance(entry, Open):
                account_map[entry.account] = entry
            elif isinstance(entry, Close):
                account_map.pop(entry.account)

        self.open_accounts = account_map

    def load_training_data(self, existing_entries):
        """Load training data, i.e., a list of Beancount entries."""
        training_data = existing_entries or []
        self.load_open_accounts(existing_entries)
        training_data = list(filter_txns(training_data))
        length_all = len(training_data)
        training_data = [
            txn for txn in training_data if self.training_data_filter(txn)
        ]
        logger.debug(
            "Filtered training data to %s of %s entries.",
            len(training_data),
            length_all,
        )
        self.training_data = training_data

    def training_data_filter(self, txn):
        """Filter function for the training data."""
        found_import_account = False
        for pos in txn.postings:
            if pos.account not in self.open_accounts:
                return False
            if self.account == pos.account:
                found_import_account = True
        return found_import_account or not self.account

    @property
    def targets(self):
        """The training targets for the given training data.

        Returns:
            A list training targets (of the same length as the training data).
        """
        if not self.attribute:
            raise NotImplementedError
        return [
            getattr(entry, self.attribute) or ""
            for entry in self.training_data
        ]

    def define_pipeline(self):
        """Defines the machine learning pipeline based on given weights."""

        transformers = []
        for attribute in self.weights:
            transformers.append((attribute, get_pipeline(attribute)))

        self.pipeline = make_pipeline(
            FeatureUnion(
                transformer_list=transformers, transformer_weights=self.weights
            ),
            SVC(kernel="linear"),
        )

    def train_pipeline(self):
        """Train the machine learning pipeline."""

        targets = self.targets
        self.is_fitted = False

        if not self.training_data:
            logger.warning(
                "Cannot train the machine learning model "
                "because the training data is empty."
            )
        elif len(set(targets)) < 2:
            logger.warning(
                "Cannot train the machine learning model "
                "because there is only one target."
            )
        else:
            self.pipeline.fit(self.training_data, targets)
            self.is_fitted = True
            logger.debug("Trained the machine learning model.")

    def process_entries(self, imported_entries) -> List[Union[ALL_DIRECTIVES]]:
        """Process imported entries.

        Transactions might be modified, all other entries are left as is.

        Returns:
            The list of entries to be imported.
        """
        enhanced_transactions = self.process_transactions(
            list(filter_txns(imported_entries))
        )
        return merge_non_transaction_entries(
            imported_entries, enhanced_transactions
        )

    def apply_prediction(self, entry, prediction):
        """Apply a single prediction to an entry.

        Args:
            entry: A Beancount entry.
            prediction: The prediction for an attribute.

        Returns:
            The entry with the prediction applied.
        """
        if not self.attribute:
            raise NotImplementedError
        return set_entry_attribute(
            entry, self.attribute, prediction, overwrite=self.overwrite
        )

    def process_transactions(
        self, transactions: List[Transaction]
    ) -> List[Transaction]:
        """Process a list of transactions."""

        if not self.is_fitted or not transactions:
            return transactions

        if self.predict:
            predictions = self.pipeline.predict(transactions)
            transactions = [
                self.apply_prediction(entry, prediction)
                for entry, prediction in zip(transactions, predictions)
            ]
            logger.debug("Added predictions to transactions.")

        return transactions
