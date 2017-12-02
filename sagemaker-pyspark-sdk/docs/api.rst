.. _api:

API
===

This part of the documentation covers all the interfaces of sagemaker_pyspark.!!!!!




SageMakerModel
------------------

.. autoclass:: sagemaker_pyspark.SageMakerModel
   :members:
   :inherited-members:
   :show-inheritance:


SageMakerEstimator
----------------------

.. autoclass:: sagemaker_pyspark.SageMakerEstimator
   :members:
   :inherited-members:
   :show-inheritance:

Algorithms
----------

K Means
^^^^^^^

.. autoclass:: sagemaker_pyspark.algorithms.KMeansSageMakerEstimator
   :members:
   :inherited-members:
   :show-inheritance:

Linear Learner Regressor
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: sagemaker_pyspark.algorithms.LinearLearnerRegressor
   :members:
   :inherited-members:
   :show-inheritance:

Linear Learner Binary Classifier
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: sagemaker_pyspark.algorithms.LinearLearnerBinaryClassifier
   :members:
   :inherited-members:
   :show-inheritance:


PCA
^^^

.. autoclass:: sagemaker_pyspark.algorithms.PCASageMakerEstimator
   :members:
   :inherited-members:
   :show-inheritance:

XGBoost
^^^^^^^

.. autoclass:: sagemaker_pyspark.algorithms.XGBoostSageMakerEstimator
   :members:
   :inherited-members:
   :show-inheritance:


Serializers
-----------

.. automodule:: sagemaker_pyspark.transformation.serializers
   :members:
   :show-inheritance:

Deserializers
-------------

.. automodule:: sagemaker_pyspark.transformation.deserializers
   :members:
   :show-inheritance:


Other Classes
--------------

.. automodule:: sagemaker_pyspark
    :members:
    :exclude-members: SageMakerModel, SageMakerEstimator, SageMakerEstimatorBase
    :show-inheritance:
