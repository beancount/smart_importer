Applying the `smart_importer` Decorators
========================================

Multiple options are available for how the `smart_importer` decorators can be applied.
The best option is to apply the decorators to subclasses of conventional, regression-tested importers,
as recommended in the README.


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


Invoking the Decorators directly from the Importer Config File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is also possible to invoke the decorators directly in your beancount importer configuration file,
like this:

.. code:: python

   from smart_importer.predict_postings import PredictPostings

   MyImporter = PredictPostings(suggest_accounts=False)(MyImporter)
   CONFIG = [
      MyImporter('whatever', 'config', 'is', 'needed')
   ]

