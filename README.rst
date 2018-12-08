smart_importer
==============

Augment
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

This package provides import hooks that can modify the imported entries. When
running the importer, the existing entries will be used as training data for a
machine learning model, which will then predict entry attributes.

The following example shows how to add the ``PredictPostings`` decorator to a
an existing CSV importer:

.. code:: python

    from beancount.ingest.importers import csv
    from beancount.ingest.importers.csv import Col

    from smart_importer import apply_hooks, PredictPostings


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


    CONFIG = [
        apply_hooks(MyBankImporter(account='Assets:MyBank:MyAccount'), [PredictPostings()])
    ]


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

Let's assume you have created an importer for "MyBank" called
``MyBankImporter``:

.. code:: python

    class MyBankImporter(importer.ImporterProtocol):
        """My existing importer"""
        # the actual importer logic would be here...

Note:
This documentation assumes you already know how to create Beancount importers.
Relevant documentation can be found under `beancount ingest
<http://furius.ca/beancount/doc/ingest>`__.  Using beancount.ingest, users can
write their own importers and use them to convert downloaded bank statements
into lists of Beancount entries.


Applying `smart_importer` hooks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Any Beancount importer can be converted into a smart importer by applying one
of the following hooks:

* ``PredictPostings`` - predict the list of postings.
* ``PredictPayees``- predict the payee of the transaction.

For example, to convert an existing ``MyBankImporter`` into a smart importer:

.. code:: python

    from your_custom_importer import MyBankImporter
    from smart_importer import apply_hooks, PredictPayees, PredictPostings

    my_bank_importer =  MyBankImporter('whatever', 'config', 'is', 'needed')
    apply_hooks(my_bank_importer, [PredictPostings(), PredictPayees()])

    CONFIG = [
        my_bank_importer,
    ]

Note that the importer hooks need to be applied to an importer instance, as
shown above.


Specifying Training Data
~~~~~~~~~~~~~~~~~~~~~~~~

The ``smart_importer`` hooks need training data, i.e. an existing list of
transactions in order to be effective. Training data can be specified by
calling bean-extract with an argument that references existing Beancount
transactions, e.g., ``bean-extract -f existing_transactions.beancount``. When
using the importer in Fava, the existing entries are used as training data
automatically.


Usage with Fava
~~~~~~~~~~~~~~~

Smart importers play nice with `Fava <https://github.com/beancount/fava>`__.
This means you can use smart importers together with Fava in the exact same way
as you would do with a conventional importer. See `Fava's help on importers
<https://github.com/beancount/fava/blob/master/fava/help/import.md>`__ for more
information.


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
