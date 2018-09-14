"""Importer decorators."""

import inspect
import logging
import operator
from functools import wraps
from typing import List, Union

from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.svm import SVC

from beancount.core.data import Transaction, ALL_DIRECTIVES, filter_txns

from smart_importer.machinelearning_helpers import load_training_data
from smart_importer.entries import merge_non_transaction_entries
from smart_importer.pipelines import PIPELINES

logger = logging.getLogger(__name__)


class ImporterDecorator():
    """Base class for Beancount importer class or method decorators.

    Instance Variables:

    * `account`: str
      The target account for the entries to be imported.
      It is obtained by calling the importer's file_account method.
    """

    def __init__(self):
        """
        Decorators that inherit from this class shall overwrite and implement
        this method, with parameters they need, for example a parameter for
        training data to be used for machine learning.
        """
        self.account = None

    def __call__(self, to_be_decorated):
        """Apply the decorator.

        It may be applied to a Beancount importer class or to its extract
        method.
        """
        if inspect.isclass(to_be_decorated):
            logger.debug('The decorator was applied to a class.')
            to_be_decorated.extract = self.patch_extract_method(
                to_be_decorated.extract)
            return to_be_decorated
        logger.debug('The decorator was applied to an instancemethod.')
        return self.patch_extract_method(to_be_decorated)

    def patch_extract_method(self, unpatched_extract):
        """Patch a Beancount importer's extract method.

        :param unpatched_extract: The importer's original extract method
        :return: A patched extract method.
        """
        decorator = self

        @wraps(unpatched_extract)
        def wrapper(self, file, existing_entries=None):

            logger.debug("Calling the importer's extract method.")
            if 'existing_entries' in inspect.signature(
                    unpatched_extract).parameters:
                imported_entries = unpatched_extract(self, file,
                                                     existing_entries)
            else:
                imported_entries = unpatched_extract(self, file)

            if not decorator.account:
                file_account = self.file_account(file)
                if file_account:
                    decorator.account = file_account
                    logger.debug(
                        f"Read file_account {file_account} from the importer; "
                        f"using it as known account in the decorator.")
                else:
                    logger.debug(
                        "Could not retrieve file_account from the importer.")

            return decorator.main(imported_entries, existing_entries)

        return wrapper

    def main(self, imported_entries, existing_entries):
        """Modify imported entries."""
        pass


class SmartImporterDecorator(ImporterDecorator):
    def __init__(self):
        super().__init__()
        self.training_data = None
        self.pipeline = None
        self.weights = {}

    def main(self, imported_entries, existing_entries):
        """Predict and suggest attributes for imported transactions."""
        try:
            self.load_training_data(existing_entries)
            self.prepare_training_data()
            self.define_pipeline()
            self.train_pipeline()
            return self.process_entries(imported_entries)
        except (ValueError, AssertionError) as exception:
            logger.error(exception)
            return imported_entries

    def load_training_data(self, existing_entries):
        """Load training data, i.e., a list of Beancount entries."""
        self.training_data = load_training_data(
            self.training_data,
            known_account=self.account,
            existing_entries=existing_entries)

    @property
    def targets(self):
        """The training targets for the given training data.

        Returns:
            A list training targets (of the same length as the training data).
        """
        raise NotImplementedError

    def prepare_training_data(self):
        """Modify the training data if necessary."""
        pass

    def define_pipeline(self):
        """Defines the machine learning pipeline based on given weights."""

        transformers = [(attribute, PIPELINES[attribute])
                        for attribute in self.weights]

        self.pipeline = make_pipeline(
            FeatureUnion(
                transformer_list=transformers,
                transformer_weights=self.weights),
            SVC(kernel='linear'),
        )

    def train_pipeline(self):
        """Train the machine learning pipeline."""

        if not self.training_data:
            raise ValueError("Cannot train the machine learning model "
                             "because the training data is empty.")
        elif len(self.training_data) < 2:
            raise ValueError(
                "Cannot train the machine learning model "
                "because the training data consists of less than two elements."
            )

        self.pipeline.fit(self.training_data, self.targets)
        logger.debug("Trained the machine learning model.")

    def process_entries(self, imported_entries) -> List[Union[ALL_DIRECTIVES]]:
        """Process imported entries.

        Transactions might be modified, all other entries are left as is.

        :return: Returns the list of entries to be imported.
        """
        imported_transactions: List[Transaction]
        imported_transactions = list(filter_txns(imported_entries))
        enhanced_transactions = self.process_transactions(
            list(imported_transactions))
        return merge_non_transaction_entries(imported_entries,
                                             enhanced_transactions)

    def apply_prediction(self, entry, prediction):
        """Apply a single prediction to an entry."""
        raise NotImplementedError

    def apply_suggestion(self, entry, suggestions):
        """Add a list of suggestions to an entry."""
        raise NotImplementedError

    def process_transactions(
            self, transactions: List[Transaction]) -> List[Transaction]:
        """Process all imported transactions."""

        if self.predict:
            predictions = self.pipeline.predict(transactions)
            transactions = [
                self.apply_prediction(entry, payee)
                for entry, payee in zip(transactions, predictions)
            ]
            logger.debug("Added predictions to transactions.")

        if self.suggest:
            # Get values from the SVC decision function
            decision_values = self.pipeline.decision_function(transactions)

            # Add a human-readable class label to each value, and sort by
            # value:
            suggestions = [[
                label for _, label in sorted(
                    list(zip(distance_values, self.pipeline.classes_)),
                    key=operator.itemgetter(0),
                    reverse=True)
            ] for distance_values in decision_values]

            # Add the suggestions to each transaction:
            transactions = [
                self.apply_suggestion(entry, suggestion_list)
                for entry, suggestion_list in zip(transactions, suggestions)
            ]
            logger.debug("Added suggestions to transactions.")
        return transactions
