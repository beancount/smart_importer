Applying the `smart_importer` Decorators
========================================

Multiple options are available for how the `smart_importer` decorators can be applied.
The best option is to apply the decorators to subclasses of conventional, regression-tested importers,
as recommended in the `README <../README.rst>`__.


Applying the Decorators to Importer Classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    @PredictPostings(training_data="trainingdata.beancount")
    @PredictPayees(training_data="trainingdata.beancount")
    class MyImporter(ImporterProtocol):
        # importer logic here...
        pass


Applying the Decorators to Extract Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    class MyImporter(ImporterProtocol):
        @PredictPostings(training_data="trainingdata.beancount")
        @PredictPayees(training_data="trainingdata.beancount")
        def extract(self, file, existing_entries):
            # importer logic here...
            pass


Applying the Decorators when Instantiating an Importer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is also possible to apply the decorators when instantiating an importer.
For example, in your import config file:


.. code:: python

    from smart_importer.predict_postings import PredictPostings
    from my_importers import MyBankImporter, MyOtherBankImporter

    # apply the decorator to define a smart importer
    SmartMyBankImporter = PredictPostings(suggest_accounts=False)(MyBankImporter)

    # use the smart importer in the config
    CONFIG = [
      MyImporter('whatever', 'config', 'is', 'needed')
    ]

    # same as above, but all in one step:
    CONFIG += [
        PredictPostings(suggest_accounts=False)(MyOtherBankImporter)('whatever', 'config')
    ]
