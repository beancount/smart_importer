smart_importer
==============

Decorators for augmenting
`Beancount <http://furius.ca/beancount/>`__ importers
with machine learning functionality.


Status
------

Working protoype, development status: alpha

.. image:: https://travis-ci.org/beancount/smart_importer.svg?branch=master
    :target: https://travis-ci.org/beancount/smart_importer


Installation
------------

The ``smart_importer`` package has not yet been published on PyPI
and must therefore be installed from source:

.. code:: bash

    git clone https://github.com/beancount/smart_importer.git
    pip install --editable smart_importer



Quick Start
-----------

Apply ``@PredictPostings()`` and/or ``@PredictPayees()`` as decorators to a Beancount importer
in order to benefit from smart predictions and suggestions provided by machine learning.
To get started quickly, you can script all of it right in your import config file.

The following example shows how to add the ``@PredictPostings`` decorator to a CSV importer:

.. code:: python

    # the beancount import config file:

    from beancount.ingest.importers import csv
    from beancount.ingest.importers.csv import Col

    from smart_importer import PredictPostings


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


    @PredictPostings()
    class SmartMyBankImporter(MyBankImporter):
        '''Smart Version of the MyBankImporter'''
        pass


    CONFIG = [
        SmartMyBankImporter(account='Assets:MyBank:MyAccount')
    ]



In the above example, the ``PredictPostings`` decorator is applied to a Beancount importer.
The resulting smart importer enhances imported transactions using machine learning.
The smart importer is added to the ``CONFIG`` array in the same way as any other Beancount importer.




Documentation
-------------

This section explains in detail the relevant concepts and artifacts
needed for enhancing Beancount importers with machine learning.



System Overview
~~~~~~~~~~~~~~~

The following figure provides an overview of the import process and its components.


.. figure:: docs/system-overview.png
   :scale: 50 %
   :alt: system overview

   System overview showing the process how smart importers are used to predict and suggest values in the transactions to be imported.


1. The user executes ``bean-extract -f existing_transactions.beancount`` in order to import downloaded bank statements into Beancount.
   Note: Instead of invoking the importer through the commandline, a user may work with a GUI such as `Fava <https://github.com/beancount/fava>`__.
2. The user specifies an import configuration file for ``bean-extract``. This file can be named, for example, ``example.import``. It is a regular python file that defines a list of importers to be used by beancount.ingest.
3. ``beancount.ingest`` invokes a matching importer.
4. The importer reads the downloaded bank statement, typically a CSV file, and extracts Beancount transactions from it.
   Note: Beancount importers are described in the `beancount ingest <http://furius.ca/beancount/doc/ingest>`__ documentation.
5. Smart importers extend the functionality of regular Beancount importers. They read existing Beancount entries and use them to train a machine learning model.
6. The smart importer uses the trained machine learning model to enhance the extracted transactions with predictions and suggestions.
7. The resulting transactions are returned to the user.


Beancount Importers
~~~~~~~~~~~~~~~~~~~~

Let's assume you have created an importer for "MyBank" called ``MyBankImporter``:

.. code:: python

    class MyBankImporter(importer.ImporterProtocol):
        """My existing importer"""
        # the actual importer logic would be here...
        pass

Note:
This documentation assumes you already know how to create Beancount importers.
Relevant documentation can be found under `beancount ingest <http://furius.ca/beancount/doc/ingest>`__.
Using beancount.ingest, users can write their own importers
and use them to convert downloaded bank statements into lists of Beancount entries.


Applying `smart_importer` Decorators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Any Beancount importer can be converted into a smart importer by applying one of the following decorators:

* ``@PredictPostings()``
* ``@PredictPayees()``

For example, to convert an existing ``MyBankImporter`` into a smart importer:

.. code:: python

    from beancount.ingest.importer import ImporterProtocol
    from smart_importer import PredictPayees
    from smart_importer import PredictPostings

    @PredictPostings()
    @PredictPayees()
    class MyBankImporter(ImporterProtocol):
        # [...] importer logic

In the above example, ``MyBankImporter`` has been decorated with
``PredictPostings`` and ``PredictPayees`` and thus employs machine learning to
predict missing payees and second postings.

Note that the decorators can be applied to either an importer class, as shown
above, or its extract method.  The result is the same in both cases.  See
`Applying the Decorators <docs/Applying_the_Decorators.rst>`__ for a
description of various alternative ways of applying the decorators to
importers.


Specifying Training Data
~~~~~~~~~~~~~~~~~~~~~~~~

The ``smart_importer`` decorators need training data, i.e. an existing list of
transactions in order to be effective. Training data can be specified by
calling bean-extract with an argument that references existing Beancount
transactions, e.g., ``bean-extract -f existing_transactions.beancount``.


Using Smart Importers
~~~~~~~~~~~~~~~~~~~~~

You can use your smart importers in the very same way as conventional importers.
I.e., you can add them to your Beancount importer configuration file, like this:

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

    from smart_importer import PredictPostings


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


Usage with Fava
~~~~~~~~~~~~~~~

Smart importers play nice with `Fava <https://github.com/beancount/fava>`__.
This means you can use smart importers together with Fava in the exact same way
as you would do with a conventional importer.
See `Fava's help on importers <https://github.com/beancount/fava/blob/master/fava/help/import.md>`__
for more information.



Development
-----------

Pull requests welcome!


Executing the Unit Tests
~~~~~~~~~~~~~~~~~~~~~~~~

Simply run (requires tox):

.. code:: bash

    make test



Configuring Logging
~~~~~~~~~~~~~~~~~~~

Python's `logging` module is used by the smart_importer decorators.
The decorators' log level can be changed as follows:


.. code:: python

    import logging

    logging.getLogger('smart_importer').setLevel(logging.DEBUG)
