import inspect
import logging
from functools import wraps
from typing import List, Union

from beancount.core.data import Transaction, ALL_DIRECTIVES, filter_txns

from smart_importer.machinelearning_helpers import load_training_data
from smart_importer.machinelearning_helpers import \
    merge_non_transaction_entries

logger = logging.getLogger(__name__)


class ImporterDecorator():
    """Abstract base class for Beancount importer class or method decorators.

    Instance Variables:

    * `existing_entries`: List[Transaction]
      Existing entries are obtained from the `existing_entries` argument that
      may be provided to a beancount importer's extract method. These entries
      are pre-existing data, typically transactions from a beancount file prior
      to starting the import.

    * `imported_entries`: List[Transaction]
      Imported entries result from calling the importer's extract method.
      These entries are the data to be imported.  Decorators that inherit from
      SmartImporterDecorator will typically modify or enhance these entries in
      some way.

    * `account`: str
      The target account for the entries to be imported.
      It is obtained by calling the importer's file_account method.
    """

    def __init__(self, training_data: List[Transaction]):
        """
        Decorators that inherit from this class shall overwrite and implement
        this method, with parameters they need, for example a parameter for
        training data to be used for machine learning.
        """
        self.training_data = training_data
        self.account = None
        self.existing_entries = None
        self.imported_entries = None

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
        assert inspect.isfunction(to_be_decorated)
        logger.debug('The decorator was applied to an instancemethod.')
        return self.patch_extract_method(to_be_decorated)

    def patch_extract_method(self, original_extract_method):
        """Patch a Beancount importer's extract method.

        :param original_extract_method: The importer's original extract method
        :return: A patched extract method.
        """
        decorator = self

        @wraps(original_extract_method)
        def wrapper(self, file, existing_entries=None):
            decorator.existing_entries = existing_entries

            logger.debug("Calling the importer's extract method.")
            if 'existing_entries' in inspect.signature(
                    original_extract_method).parameters:
                decorator.imported_entries = original_extract_method(
                    self, file, existing_entries)
            else:
                decorator.imported_entries = original_extract_method(
                    self, file)

            if not decorator.account:
                logger.debug(
                    "Trying to read the importer's file_account, "
                    "to be used as default value for the decorator's "
                    "`account` argument..."
                )
                file_account = self.file_account(file)
                if file_account:
                    decorator.account = file_account
                    logger.debug(
                        f"Read file_account {file_account} from the importer; "
                        f"using it as known account in the decorator.")
                else:
                    logger.debug(
                        "Could not retrieve file_account from the importer.")

            return decorator.main()

        return wrapper

    def main(self) -> List[Union[ALL_DIRECTIVES]]:
        """Predict and suggest attributes for imported transactions."""
        pass


class SmartImporterDecorator(ImporterDecorator):
    def main(self) -> List[Union[ALL_DIRECTIVES]]:
        """Predict and suggest attributes for imported transactions."""
        try:
            self.load_training_data()
            self.prepare_training_data()
            self.define_pipeline()
            self.train_pipeline()
            return self.process_entries()
        except (ValueError, AssertionError) as exception:
            logger.error(exception)
            return self.imported_entries

    def load_training_data(self):
        """Load training data, i.e., a list of Beancount entries."""
        self.training_data = load_training_data(
            self.training_data,
            known_account=self.account,
            existing_entries=self.existing_entries)

    def prepare_training_data(self):
        pass

    def define_pipeline(self):
        raise NotImplementedError

    def train_pipeline(self):
        raise NotImplementedError

    def process_entries(self) -> List[Union[ALL_DIRECTIVES]]:
        """Process imported entries.

        Transactions are enhanced, all other entries are left as is.

        :return: Returns the list of entries to be imported.
        """
        imported_transactions: List[Transaction]
        imported_transactions = list(filter_txns(self.imported_entries))
        enhanced_transactions = self.process_transactions(
            list(imported_transactions))
        return merge_non_transaction_entries(self.imported_entries,
                                             enhanced_transactions)

    def process_transactions(self, transactions):
        raise NotImplementedError
