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

The `smart_importer` package has not yet been published on PyPI
and must therefore be installed from source:

.. code:: bash

    git clone https://github.com/beancount/smart_importer.git
    cd smart_importer
    pip install --editable .



Quick Start
-----------

Apply `@PredictPostings()` and/or `@PredictPayees()` as decorators to a beancount importer
in order to benefit from smart predictions and suggestions provided by machine learning:


.. code:: python

    from beancount.ingest.importer import ImporterProtocol
    from smart_importer.predict_postings import PredictPostings

    class MyImporter(ImporterProtocol):
        def extract(self, file, existing_entries):
          # do the import, e.g., from a csv file

    @PredictPostings(training_data='myfile.beancount')
    class MySmartImporter(MyImporter):
        pass


In the above example, the `PredictPostings` decorator from `smart_importer` is applied to a beancount importer.
The resulting `MySmartImporter` importer enhances imported transactions using machine learning.
The machine learning algorithm uses training data from an existing beancount file.

You can use the smart (i.e., decorated) importer in the exact same way as you would do with a regular importer.
For example, in your beancount import configuration file:

.. code:: python

    CONFIG = [
        MyImporter('whatever', 'config', 'is', 'needed')
    ]



Usage
-----

This section guides through the creation of a smart beancount importer.


Conventional Beancount Importers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This documentation assumes you have existing beancount importers,
as described in beancount's `documentation <http://furius.ca/beancount/doc/index>`__.
For example, an importer called `MyImporter`:

.. code:: python

    class MyImporter(importer.ImporterProtocol):
        """My existing importer"""
        # the actual importer logic would be here...
        pass


Applying `smart_importer` Decorators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Any beancount importer can be converted into a smart importer by applying one of the following decorators:

* `@PredictPostings()`
* `@PredictPayees()`


For example:

.. code:: python

    from beancount.ingest.importer import ImporterProtocol
    from smart_importer.predict_postings import PredictPostings
    from smart_importer.predict_postings import PredictPayees

    class MyImporter(ImporterProtocol):
        def extract(self, file, existing_entries):
          # do the import, e.g., from a csv file

    @PredictPostings()
    @PredictPayees()
    class MySmartImporter(MyImporter):
        pass


Note that the decorators can be applied to either an importer class, as shown above, or its extract method.
In both cases, the result is the same.


Specifying Training Data
~~~~~~~~~~~~~~~~~~~~~~~~

The `smart_importer` decorators must be fed with training data in order to be effective.

Training data can be provided directly as an argument to the decorators.
You can simply provide the name of your beancount file, like this:

.. code:: python

    @PredictPostings(training_data='file.beancount')


If no training data is explicitly provided as an argument,
the decorators try to use the `existing_entries` that can be passed to an importer's `extract` method.



Using Smart Importers
~~~~~~~~~~~~~~~~~~~~~

Once you have decorated your importers (or new subclasses thereof, see the below section on unit testing),
you can start using your smart importers in the same way as conventional importers.
I.e., you can add them to your beancount importer configuration file, like this:

.. code:: python

   CONFIG = [
      MySmartImporter('whatever', 'config', 'is', 'needed')
   ]



Unit Testing your Importers
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Smart importers are difficult to unit-test because their output depends on dynamic machine learning behavior.
To make unit testing easy, you can continue to simply write your unit tests for conventional (undecorated) importers
if you apply the decorators to subclasses, like this:


.. code:: python

    # The existing, conventional importer class
    class MyImporter(importer.ImporterProtocol):
        """My existing importer, without machine learning functionality, left undecorated to ease unit testing"""
        # the actual importer logic would be here...
        pass


    # Apply the decorator to a new subclass of your importer:
    @PredictPostings()
    class MySmartImporter(MyConventionalImporter):
        """
        The smart version of my existing, conventional importer,
        ready to be used in your import configuration file.
        """
        pass


    # MyImporter can be unit-tested,
    # e.g., using `beancount.ingest.regression.compare_sample_files`:
    def test():
        importer = MyConventionalImporter()
        yield from regression.compare_sample_files(
            importer,
            directory=os.path.abspath(os.path.join(
                os.path.dirname(__file__), 'testdata'))
        )

    if __name__ == "__main__":
        nose.main(config=Config())




Development
-----------

Pull requests welcome!


.. code:: bash

    # for nicer test output:
    pip install coloredlogs

    # to run unittests:
    make test
