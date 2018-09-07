import inspect
import logging
from functools import wraps
from typing import List, Union

from beancount.core.data import Transaction, ALL_DIRECTIVES, filter_txns

from smart_importer.machinelearning_helpers import load_training_data
from smart_importer.entries import merge_non_transaction_entries

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
        assert inspect.isfunction(to_be_decorated)
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

            return decorator.main(imported_entries, existing_entries)

        return wrapper

    def main(self, imported_entries, existing_entries):
        """Modify imported entries."""
        pass


class SmartImporterDecorator(ImporterDecorator):
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

    def prepare_training_data(self):
        pass

    def define_pipeline(self):
        raise NotImplementedError

    def train_pipeline(self):
        raise NotImplementedError

    def process_entries(self, imported_entries) -> List[Union[ALL_DIRECTIVES]]:
        """Process imported entries.

        Transactions are enhanced, all other entries are left as is.

        :return: Returns the list of entries to be imported.
        """
        imported_transactions: List[Transaction]
        imported_transactions = list(filter_txns(imported_entries))
        enhanced_transactions = self.process_transactions(
            list(imported_transactions))
        return merge_non_transaction_entries(imported_entries,
                                             enhanced_transactions)

    def process_transactions(self, transactions):
        raise NotImplementedError
