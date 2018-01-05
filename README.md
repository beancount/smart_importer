# smart_importer

A smart importer for beancount, built as a suggestion for https://github.com/beancount/fava/issues/579


Todos:

* improve unit tests for suggested and predicted accounts
* differentiate between predictions and suggestions being made:
  * predicting account names means that the decorator adds predicted second postings to the import
  * suggesting account names leaves the import data untouched, but only adds `__completions__` metadata, leaving it up to the user to add second postings.
* add another decorator that predicts payees
* fix handling of imported statements that are not transactions (but, for example, balance assertions)
