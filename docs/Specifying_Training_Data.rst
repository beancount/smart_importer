Specifying Training Data
========================

Multiple options are available for specifying training data to a ``smart_importer`` decorator.
The training data is used to train the machine learning models that enhance the imported transactions.


Existing Entries Passed to ``bean-extract``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The easiest and recommended option is to specifiy an existing beancount file when calling bean-extract,
e.g., ``bean-extract -f existing_transactions.beancount``.
The decorators will use existing transactions from this file as training data.



Existing Entries Passed to the Decorator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Existing entries can be specified as an argument ``training_data`` like this:

.. code:: python

    @PredictPostings(training_data='ledger.beancount')

