smart_importer
==============

Augment
`Beancount <http://furius.ca/beancount/>`__ importers
with machine learning functionality.


Status
------

Working protoype, development status: beta


Installation
------------

The ``smart_importer`` can be installed from PyPI:

.. code:: bash

    pip install smart_importer


Quick Start
-----------

This package provides import hooks that can modify the imported entries. When
running the importer, the existing entries will be used as training data for a
machine learning model, which will then predict entry attributes.

The following example shows how to apply the ``PredictPostings`` hook to
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
                (
                    'Date, Transaction Details, Funds Out, Funds In'
                )
            )


    CONFIG = [
        apply_hooks(MyBankImporter(account='Assets:MyBank:MyAccount'), [PredictPostings()])
    ]


Documentation
-------------

This section explains in detail the relevant concepts and artifacts
needed for enhancing Beancount importers with machine learning.


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
<https://beancount.github.io/docs/17_importing_external_data.html>`__.  Using beancount.ingest, users can
write their own importers and use them to convert downloaded bank statements
into lists of Beancount entries.
An example is provided as part of beancount's source code under
`beancount/ingest/office
<https://github.com/beancount/beancount/tree/master/examples/ingest/office>`__.

smart_importer only works by appending onto incomplete single-legged postings (i.e. It will not work by modifying postings with accounts like "Expenses:TODO"). The `extract` method in the importer should follow the `latest interface <https://github.com/beancount/beancount/blob/master/beancount/ingest/importer.py#L61>`__ and include an `existing_entries` argument. 

Applying `smart_importer` hooks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Any Beancount importer can be converted into a smart importer by applying one
of the following hooks:

* ``PredictPostings`` - predict the list of postings.
* ``PredictPayees``- predict the payee of the transaction.
* ``DuplicateDetector`` - detect duplicates

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
<https://beancount.github.io/fava/api/beancount.ingest.html#module-beancount.ingest.importer>`__ for more
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

Python's `logging` module is used by the smart_importer module.
The according log level can be changed as follows:


.. code:: python

    import logging
    logging.getLogger('smart_importer').setLevel(logging.DEBUG)
