smart_importer
==============

A smart importer for
`beancount <https://github.com/beancount/beancount>`__ and
`fava <https://github.com/beancount/fava>`__, currently in development
as a suggestion for `#579 Import: Intelligent suggestions for account
names <https://github.com/beancount/fava/issues/579>`__


Status
------

Prototype, work in progress.

.. image:: https://travis-ci.org/johannesjh/smart_importer.svg?branch=master
   :target: https://travis-ci.org/johannesjh/smart_importer


Installation
------------

.. code:: bash

    pip install smart_importer

    # or, to install from local git clone:
    pip install --editable .



Current Functionality
---------------------

When writing a beancount importer, users can apply decorators to their
importer’s ``extract`` function in order to benefit from predictions and
suggestions provided by machine learning.

For example:

.. code:: python

    class MyImporter(ImporterProtocol):
            @PredictPostings(training_data="trainingdata.beancount")
            @PredictPayees(training_data="trainingdata.beancount")
            def extract(file):
              # do the import, e.g., from a csv file


Todos:
------

-  [x] cleanup and simplify the conversion from beancount entries to
   scikit-learn feature vectors. E.g. by implementing a
   `FunctionTransformer <http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html#sklearn.preprocessing.FunctionTransformer>`__
   that converts
   `TxnPosting <https://aumayr.github.io/beancount-docs-static/api_reference/beancount.core.html?highlight=txnposting#beancount.core.data.TxnPosting>`__\ s
   to feature vectors and back.
-  [x] prepare for additional decorators: some refactoring and cleanup
   is needed to ease the implementation of additional decorators.
-  [x] add another decorator that predicts payees
-  [x] fix handling of imported statements that are not transactions
   (but, for example, balance assertions), see issue #1
-  [ ] package as pip package
-  [ ] limit predictions to missing second postings (since it probably
   does not make sense to predict third and fourth postings)
-  [ ] add unittests to ensure the two decorators play together nicely
-  [ ] stability: add a unit test and gracefully handle the case for
   when there are only two accounts in the training data (this currently
   throws an error)

-  [ ] clarify and decide packaging and integration with beancount
   and/or fava

