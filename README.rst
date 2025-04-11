smart_importer
==============

https://github.com/beancount/smart_importer

.. image:: https://github.com/beancount/smart_importer/actions/workflows/ci.yml/badge.svg?branch=main
   :target: https://github.com/beancount/smart_importer/actions?query=branch%3Amain

Augments
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

    from beangulp.importers import csv
    from beangulp.importers.csv import Col

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
                (
                    'Date, Transaction Details, Funds Out, Funds In'
                )
            )


    CONFIG = [
        MyBankImporter(account='Assets:MyBank:MyAccount'),
    ]

    HOOKS = [
        PredictPostings().hook
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

    class MyBankImporter(importer.Importer):
        """My existing importer"""
        # the actual importer logic would be here...

Note:
This documentation assumes you already know how to create Beancount/Beangulp importers.
Relevant documentation can be found in the `beancount import documentation
<https://beancount.github.io/docs/importing_external_data.html>`__.
With the functionality of beangulp, users can
write their own importers and use them to convert downloaded bank statements
into lists of Beancount entries.
Examples are provided as part of beangulps source code under
`examples/importers
<https://github.com/beancount/beangulp/tree/master/examples/importers>`__.

smart_importer only works by appending onto incomplete single-legged postings
(i.e. It will not work by modifying postings with accounts like "Expenses:TODO").
The `extract` method in the importer should follow the
`latest interface <https://github.com/beancount/beangulp/blob/master/beangulp/importer.py>`__
and include an `existing_entries` argument.

Using `smart_importer` as a beangulp hook
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Beangulp has the notation of hooks, for some detailed example see `beangulp hook example <https://github.com/beancount/beangulp/blob/ead8a2517d4f34c7ac7d48e4ef6d21a88be7363c/examples/import.py#L50>`.
This can be used to apply smart importer to all importers.

* ``PredictPostings`` - predict the list of postings.
* ``PredictPayees``- predict the payee of the transaction.

For example, to convert an existing ``MyBankImporter`` into a smart importer:

.. code:: python

    from your_custom_importer import MyBankImporter
    from smart_importer import PredictPayees, PredictPostings

    CONFIG = [
        MyBankImporter('whatever', 'config', 'is', 'needed'),
    ]

    HOOKS = [
        PredictPostings().hook,
        PredictPayees().hook
    ]

Wrapping an importer to become a  `smart_importer`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instead of using a beangulp hook, it's possible to wrap any importer to become a smart importer, this will modify only this importer.

* ``PredictPostings`` - predict the list of postings.
* ``PredictPayees``- predict the payee of the transaction.

For example, to convert an existing ``MyBankImporter`` into a smart importer:

.. code:: python

    from your_custom_importer import MyBankImporter
    from smart_importer import PredictPayees, PredictPostings

    CONFIG = [
        PredictPostings().wrap(
            PredictPayees().wrap(
                MyBankImporter('whatever', 'config', 'is', 'needed')
            )
        ),
    ]

    HOOKS = [
    ]


Specifying Training Data
~~~~~~~~~~~~~~~~~~~~~~~~

The ``smart_importer`` hooks need training data, i.e. an existing list of
transactions in order to be effective. Training data can be specified by
calling bean-extract with an argument that references existing Beancount
transactions, e.g., ``import.py extract -e existing_transactions.beancount``. When
using the importer in Fava, the existing entries are used as training data
automatically.


Usage with Fava
~~~~~~~~~~~~~~~

Smart importers play nice with `Fava <https://github.com/beancount/fava>`__.
This means you can use smart importers together with Fava in the exact same way
as you would do with a conventional importer. See `Fava's help on importers
<https://github.com/beancount/fava/blob/main/src/fava/help/import.md>`__ for more
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


Using Tokenizer
~~~~~~~~~~~~~~~~~~

Custom tokenizers can let smart_importer support more languages, eg. Chinese.

If you looking for Chinese tokenizer, you can follow this example:

First make sure that `jieba` is installed in your python environment:

.. code:: bash

    pip install jieba


In your importer code, you can then pass `jieba` to be used as tokenizer:

.. code:: python

    from smart_importer import PredictPostings
    import jieba

    jieba.initialize()
    tokenizer = lambda s: list(jieba.cut(s))

    predictor = PredictPostings(string_tokenizer=tokenizer)
