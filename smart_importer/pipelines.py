"""Machine learning pipelines for data extraction."""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

from smart_importer import machinelearning_helpers as ml


PIPELINES = {
    'narration': Pipeline([
        ('get_narration', ml.GetNarration()),
        ('vect', CountVectorizer(ngram_range=(1, 3))),
    ]),
    'payee': Pipeline([
        ('get_payee', ml.GetPayee()),
        ('vect', CountVectorizer(ngram_range=(1, 3))),
    ]),
    'first_posting_account': Pipeline([
        ('get_first_posting_account', ml.GetReferencePostingAccount()),
        ('vect', CountVectorizer(ngram_range=(1, 3))),
    ]),
    'date.day': Pipeline([
        ('get_date.day', ml.GetDayOfMonth()),
        ('caster', ml.ArrayCaster()),
    ]),
}
