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



Usage
-----

When writing a beancount importer, users can apply decorators to their importer classes
in order to benefit from smart predictions and suggestions provided by machine learning.

For example:

.. code:: python

    @PredictPostings(training_data="trainingdata.beancount")
    @PredictPayees(training_data="trainingdata.beancount")
    class MyImporter(ImporterProtocol):
        def extract(file):
          # do the import, e.g., from a csv file

If you don't want to modify your importers you can also manually instantiate this 
e.g. in our foo.import

.. code:: python

   from smart_importer.predict_postings import PredictPostings

   MyImporter = PredictPostings(suggest_accounts=False)(MyImporter)
   CONFIG = [
      MyImporter("someconfig")
   ]


Development
-----------

.. code:: bash

    # for nicer test output:
    pip install coloredlogs

    # to run unittests:
    make test
