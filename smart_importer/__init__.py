"""Smart importer for Beancount and Fava."""

__version__ = '0.0.1'
__copyright__ = "Copyright (C) 2018 Johannes Harms"
__license__ = "MIT"

from smart_importer.entries import add_payee_to_transaction
from smart_importer.entries import add_suggested_payees_to_transaction
from smart_importer.predict_postings import PredictPostings  # noqa
from smart_importer.predictor import SmartImporterDecorator


class PredictPayees(SmartImporterDecorator):
    """Suggest and predict payees."""

    def __init__(self, predict=True, overwrite=False, suggest=False):
        super().__init__()

        self.predict = predict
        self.overwrite = overwrite
        self.suggest = suggest

        self.weights = {'narration': 0.8, 'payee': 0.5, 'date.day': 0.1}

    @property
    def targets(self):
        return [txn.payee or '' for txn in self.training_data]

    def apply_prediction(self, entry, prediction):
        return add_payee_to_transaction(entry, prediction,
                                        overwrite=self.overwrite)

    def apply_suggestion(self, entry, suggestions):
        return add_suggested_payees_to_transaction(entry, suggestions)
