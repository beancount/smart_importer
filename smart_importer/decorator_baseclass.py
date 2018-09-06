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

    # Implementation notes for how to write decorators for classes, see e.g.,
    # https://stackoverflow.com/a/9910180
    # https://www.codementor.io/sheena/advanced-use-python-decorators-class-function-du107nxsv
    # https://andrefsp.wordpress.com/2012/08/23/writing-a-class-decorator-in-python/

    def __call__(self, to_be_decorated):
        """
        This method is called when the decorator is applied. It may be applied
        to a Beancount importer class or to its extract method.
        """
        if inspect.isclass(to_be_decorated):
            logger.debug('The Decorator was applied to a class.')
            return self.patched_importer_class(to_be_decorated)
        elif inspect.isfunction(to_be_decorated):
            logger.debug('The Decorator was applied to an instancemethod.')
            return self.patched_extract_method(to_be_decorated)

    def patched_importer_class(self, importer_class):
        """
        Patches a beancount importer class by modifying its extract method.
        :param importer_class: The original importer class
        :return: The modified importer class with a patched extract method.
        """
        importer_class.extract = self.patched_extract_method(
            importer_class.extract)
        return importer_class

    def patched_extract_method(self, original_extract_method):
        """
        Patches a beancount importer's extract method by wrapping it.
        :param original_extract_method: The importer's original extract method
        :return: A patched extract method, created by wrapping the original
            extract method.
        """
        decorator = self

        @wraps(original_extract_method)
        def wrapper(self, file, existing_entries=None):

            # read the importer's existing entries, if provided as argument to
            # its `extract` method:
            decorator.existing_entries = existing_entries

            # read the importer's `extract`ed entries
            logger.debug("About to call the importer's extract method to "
                         "receive entries to be imported...")
            if 'existing_entries' in inspect.signature(
                    original_extract_method).parameters:
                decorator.imported_entries = original_extract_method(
                    self, file, existing_entries)
            else:
                decorator.imported_entries = original_extract_method(
                    self, file)

            # read the importer's file_account, to be used as default value for
            # the decorator's known `account`:
            if inspect.ismethod(self.file_account) and not decorator.account:
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
                        f"Could not retrieve file_account from the importer.")

            return decorator.main()

        return wrapper

    def main(self) -> List[Union[ALL_DIRECTIVES]]:
        """
        The decorator's main method, to be implemented by inheriting classes
        with the following functionality:

        1. read `self.imported_entries`
        2. process these entries
        3. return possibly modified entries
        """
        pass


class SmartImporterDecorator(ImporterDecorator):
    def main(self):
        """
        The decorator's main method predicts and suggests the account names
        for imported transactions.
        """
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
        """Loads training data, i.e., a list of Beancount entries."""
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
        """
        Processes all imported entries (transactions as well as other types of
        entries).  Transactions are enhanced, but all other entries are left as
        is.
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
