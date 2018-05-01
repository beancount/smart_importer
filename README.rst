smart_importer
==============

A smart importer for
`beancount <https://github.com/beancount/beancount>`__ and
`fava <https://github.com/beancount/fava>`__, currently in development
as a suggestion for `#579 Import: Intelligent suggestions for account
names <https://github.com/beancount/fava/issues/579>`__


Status
------

First working protoype,
development status: alpha

.. image:: https://travis-ci.org/beancount/smart_importer.svg?branch=master
    :target: https://travis-ci.org/beancount/smart_importer


Installation
------------

The ``smart_importer`` package has not yet been published on PyPI
and must therefore be installed from source:

.. code:: bash

    git clone https://github.com/beancount/smart_importer.git
    cd smart_importer
    pip install --editable .



Quick Start
-----------

Apply ``@PredictPostings()`` and/or ``@PredictPayees()`` as decorators to a beancount importer
in order to benefit from smart predictions and suggestions provided by machine learning.
To get started quickly, you can script all of it right in your import config file.

The following example shows how to add the ``@PredictPostings`` decorator to a CSV importer:

.. code:: python

    # the beancount import config file:

    from beancount.ingest.importers import csv
    from beancount.ingest.importers.csv import Col

    from smart_importer.predict_postings import PredictPostings


    class MyBankImporter(csv.Importer):
        '''Conventional importer for MyBank'''

        def __init__(self, *, account):
            super().__init__(
                {Col.DATE: 'Date',
                 Col.PAYEE: 'Transaction Details',
                 Col.AMOUNT_DEBIT: 'Funds Out',
                 Col.AMOUNT_CREDIT: 'Funds In'},
                account,
                'EUR',
                [
                    'Filename: .*MyBank.*\.csv',
                    'Contents:\n.*Date, Transaction Details, Funds Out, Funds In'
                ]
            )


    @PredictPostings(training_data='myledger.beancount')
    class SmartMyBankImporter(MyBankImporter):
        '''Smart Version of the MyBankImporter'''
        pass


    CONFIG = [
        SmartMyBankImporter(account='Assets:MyBank:MyAccount')
    ]



In the above example, the ``PredictPostings`` decorator is applied to a beancount importer.
The resulting smart importer enhances imported transactions using machine learning.
The smart importer is added to the ``CONFIG`` array in the same way as any other beancount importer.



System Overview
---------------

The following figure provides an overview of the import process and its components.


.. figure:: docs/system-overview.png
   :scale: 50 %
   :alt: system overview

   System overview showing the process how smart importers are used to predict and suggest values in the transactions to be imported.


1. The user executes ``bean-extract`` in order to import downloaded bank statements into beancount.
2. The user must specify an import configuration file for ``bean-extract``. This file defines a list of importers to be used by beancount.ingest.
3. ```beancount.ingest`` invokes a matching importer.
4. The importer reads the downloaded bank statement, typically a CSV file, and extracts beancount transactions from it.
5. Smart importers read existing beancount entries and use them to train a machine learning model.
6. Smart importers use the trained machine learning model to enhance the extracted transactions with predictions and suggestions.
7. The resulting transactions are returned to the user.



Usage
-----

This section explains relevant concepts and artifacts
and guides through the creation of a smart beancount importer.


Beancount Importers
~~~~~~~~~~~~~~~~~~~~

The documentation on `beancount ingest <http://furius.ca/beancount/doc/ingest>`__
describes how users can write their own importers
and use them to convert downloaded bank statements into lists of beancount transactions.

This documentation assumes you have created beancount importers already.
For example, an importer for "MyBank" called ``MyBankImporter``:

.. code:: python

    class MyBankImporter(importer.ImporterProtocol):
        """My existing importer"""
        # the actual importer logic would be here...
        pass


Applying `smart_importer` Decorators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Any beancount importer can be converted into a smart importer by applying one of the following decorators:

* ``@PredictPostings()``
* ``@PredictPayees()``


For example, to convert an existing ``MyBankImporter`` into a smart importer:

.. code:: python

    from beancount.ingest.importer import ImporterProtocol
    from smart_importer.predict_postings import PredictPostings
    from smart_importer.predict_postings import PredictPayees

    class MyBankImporter(ImporterProtocol):
        def extract(self, file, existing_entries):
          # do the import, e.g., from a csv file

    @PredictPostings()
    @PredictPayees()
    class SmartMyBankImporter(MyImporter):
        pass

In the above example, ``SmartMyBankImporter`` has been decorated with ``@PredictPostings``
and thus employs machine learnign to predict missing second postings.

Note that the decorators can be applied to either an importer class, as shown above, or its extract method.
The result is the same in both cases.
See `Applying the Decorators <docs/Applying_the_Decorators.rst>`__
for a description of various ways how the decorators can be applied to importers.



Specifying Training Data
~~~~~~~~~~~~~~~~~~~~~~~~

The ``smart_importer`` decorators must be fed with training data in order to be effective.

Training data can be provided directly as an argument ``training_data`` to the decorators:

.. code:: python

    @PredictPostings(training_data='ledger.beancount')


If training data is not provided as an argument,
the decorators try to use the ``existing_entries`` that can be passed to an importer's ``extract`` method.



Using Smart Importers
~~~~~~~~~~~~~~~~~~~~~

You can use your smart importers in the very same way as conventional importers.
I.e., you can add them to your beancount importer configuration file, like this:

.. code:: python

   CONFIG = [
      MySmartImporter('whatever', 'config', 'is', 'needed')
   ]



Unit Testing your Importers
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Smart importers are difficult to unit-test because their output depends on dynamic machine learning behavior.
To make test automation easy, write unit tests for conventional (undecorated) importers,
but use decorated versions of these importers in your import configuration:


.. code:: python

    import os

    import nose
    from beancount.ingest import regression
    from beancount.ingest.importers import csv
    from beancount.ingest.importers.csv import Col

    from smart_importer.predict_postings import PredictPostings


    # define a conventional (i.e., undecorated) importer:
    class MyBankImporter(csv.Importer):
        '''
        Importer CSV file downloaded from MyBank.
        Note: This undecorated class can be regression-tested with
        beancount.ingest.regression.compare_sample_files
        '''

        def __init__(self, *, account):
            super().__init__(
                {Col.DATE: 'Date',
                 Col.PAYEE: 'Transaction Details',
                 Col.AMOUNT_DEBIT: 'Funds Out',
                 Col.AMOUNT_CREDIT: 'Funds In'},
                account,
                'CAD',
                [
                    'Filename: .*MyBank.*\.csv',
                    'Contents:\n.*Date, Transaction Details, Funds Out, Funds In'
                ]
            )


    # automated regression tests for the undecorated importer:
    def test():
        importer = MyBankImporter()
        yield from regression.compare_sample_files(
            importer,
            directory=os.path.abspath(os.path.join(
                os.path.dirname(__file__), 'testdata'))
        )


    # execute regression tests if this is run as main python file:
    if __name__ == "__main__":
        nose.main()


    # define a smart version of the importer:
    @PredictPostings(training_data='myfile.beancount')
    class SmartMyBankImporter(MyBankImporter):
        '''Smart version of MyBankImporter'''
        pass


    # the import configuration:
    CONFIG = [
        SmartMyBankImporter(account='Assets:MyBank:MyAccount')
    ]


Usage with fava
~~~~~~~~~~~~~~~

Smart importers play nice with `fava <https://github.com/beancount/fava>`__.
This means you can use smart importers together with fava in the exact same way
as you would do with a conventional importer.
See `fava's help on importers <https://github.com/beancount/fava/blob/master/fava/help/import.md>`__
for more information.



Development
-----------

Pull requests welcome!


.. code:: bash

    # for nicer test output:
    pip install coloredlogs

    # to run unittests:
    make test
