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

