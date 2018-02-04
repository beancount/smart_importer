# smart_importer

A smart importer for beancount, built as a suggestion for https://github.com/beancount/fava/issues/579


Todos:

- [x] cleanup and simplify the conversion from beancount entries to scikit-learn feature vectors. E.g. by implementing a [FunctionTransformer](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html#sklearn.preprocessing.FunctionTransformer) that converts [TxnPosting](https://aumayr.github.io/beancount-docs-static/api_reference/beancount.core.html?highlight=txnposting#beancount.core.data.TxnPosting)s to feature vectors and back.
- [x] prepare for additional decorators: some refactoring and cleanup is needed to ease the implementation of additional decorators.
- add another decorator that predicts payees

- stability: fix handling of imported statements that are not transactions (but, for example, balance assertions)
- stability: add a unit test and gracefully handle the case for when there are only two accounts in the training data (this currently throws an error)
