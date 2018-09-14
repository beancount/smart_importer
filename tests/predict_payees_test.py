"""Tests for the `PredictPayees` decorator"""

from beancount.ingest.importer import ImporterProtocol
from beancount.parser import parser

from smart_importer.entries import METADATA_KEY_SUGGESTED_PAYEES
from smart_importer.predict_payees import PredictPayees


TEST_DATA, _, __ = parser.parse_string("""
2017-01-06 * "Farmer Fresh" "Buying groceries"
  Assets:US:BofA:Checking  -2.50 USD

2017-01-07 * "Groceries"
  Assets:US:BofA:Checking  -10.20 USD

2017-01-10 * "" "Eating out with Joe"
  Assets:US:BofA:Checking  -38.36 USD

2017-01-10 * "Dinner with Martin"
  Assets:US:BofA:Checking  -35.00 USD

2017-01-10 * "Groceries"
  Assets:US:BofA:Checking  -53.70 USD

2017-01-10 * "Gimme Coffee" "Coffee"
  Assets:US:BofA:Checking  -5.00 USD
""")


TRAINING_DATA, _, __ = parser.parse_string("""
2016-01-06 * "Farmer Fresh" "Buying groceries"
  Assets:US:BofA:Checking  -2.50 USD
  Expenses:Food:Groceries

2016-01-07 * "Starbucks" "Coffee"
  Assets:US:BofA:Checking  -4.00 USD
  Expenses:Food:Coffee

2016-01-07 * "Farmer Fresh" "Groceries"
  Assets:US:BofA:Checking  -10.20 USD
  Expenses:Food:Groceries

2016-01-07 * "Gimme Coffee" "Coffee"
  Assets:US:BofA:Checking  -3.50 USD
  Expenses:Food:Coffee

2016-01-08 * "Uncle Boons" "Eating out with Joe"
  Assets:US:BofA:Checking  -38.36 USD
  Expenses:Food:Restaurant

2016-01-10 * "Walmarts" "Groceries"
  Assets:US:BofA:Checking  -53.70 USD
  Expenses:Food:Groceries

2016-01-10 * "Gimme Coffee" "Coffee"
  Assets:US:BofA:Checking  -6.19 USD
  Expenses:Food:Coffee

2016-01-10 * "Uncle Boons" "Dinner with Mary"
  Assets:US:BofA:Checking  -35.00 USD
  Expenses:Food:Restaurant

2016-01-11 * "Farmer Fresh" "Groceries"
  Assets:US:BofA:Checking  -30.50 USD
  Expenses:Food:Groceries
""")

CORRECT_PREDICTIONS = [
    'Farmer Fresh', 'Farmer Fresh', 'Uncle Boons', 'Uncle Boons',
    'Farmer Fresh', 'Gimme Coffee'
]


class BasicTestImporter(ImporterProtocol):
    def extract(self, file, existing_entries=None):
        return TEST_DATA

    def file_account(self, file):
        return "Assets:US:BofA:Checking"

@PredictPayees(suggest=True)
class DecoratedTestImporter(BasicTestImporter):
    pass

IMPORTER = DecoratedTestImporter()

def test_unchanged_narrations():
    """
    Verifies that the decorator leaves the narration intact
    """
    correct_narrations = [
        transaction.narration for transaction in TEST_DATA
    ]
    extracted_narrations = [
        transaction.narration
        for transaction in IMPORTER.extract("dummy-data",
                                            existing_entries=TRAINING_DATA)
    ]
    assert extracted_narrations == correct_narrations

def test_unchanged_first_posting():
    """
    Verifies that the decorator leaves the first posting intact
    """
    correct_first_postings = [
        transaction.postings[0] for transaction in TEST_DATA
    ]
    extracted_first_postings = [
        transaction.postings[0]
        for transaction in IMPORTER.extract("dummy-data",
                                            existing_entries=TRAINING_DATA)
    ]
    assert extracted_first_postings == correct_first_postings

def test_predicted_payees():
    """
    Verifies that the decorator adds predicted postings.
    """
    transactions = IMPORTER.extract("dummy-data",
                                    existing_entries=TRAINING_DATA)
    predicted_payees = [transaction.payee for transaction in transactions]
    assert predicted_payees == CORRECT_PREDICTIONS

def test_added_suggestions():
    """
    Verifies that the decorator adds suggestions about accounts
    that are likely to be involved in the transaction.
    """
    transactions = IMPORTER.extract("dummy-data",
                                    existing_entries=TRAINING_DATA)
    for transaction in transactions:
        assert transaction.meta[METADATA_KEY_SUGGESTED_PAYEES]
