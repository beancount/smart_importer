# smart_importer

A smart importer for beancount, built as a suggestion for https://github.com/beancount/fava/issues/579


Todos:

* add another decorator that predicts payees
* stability:
  * fix handling of imported statements that are not transactions (but, for example, balance assertions)
  * add a unit test and gracefully handle the case for when there are only two accounts in the training data (this currently throws an error)
