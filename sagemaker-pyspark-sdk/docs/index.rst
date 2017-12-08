Amazon SageMaker PySpark Documentation
======================================

The SageMaker PySpark SDK provides a pyspark interface to Amazon SageMaker, allowing customers to
train using the Spark Estimator API, host their model on Amazon SageMaker, and make predictions
with their model using the Spark Transformer API. This page is a quick guide on the basics of
SageMaker PySpark. You can also check the :ref:`api` docs.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

Quick Start
------------

First, install the library:

.. code-block:: sh

    $ pip install sagemaker_pyspark

Next, set up credentials (in e.g. ``~/.aws/credentials``):

.. code-block:: ini

    [default]
    aws_access_key_id = YOUR_KEY
    aws_secret_access_key = YOUR_KEY

Then, set up a default region (in e.g. ``~/.aws/config``):

.. code-block:: ini

    [default]
    region=us-west-2

Then, to load the sagemaker jars programatically:

.. code-block:: python

    from pyspark import SparkContext, SparkConf
    import sagemaker_pyspark

    conf = (SparkConf()
            .set("spark.driver.extraClassPath", ":".join(sagemaker_pyspark.classpath_jars())))
    SparkContext(conf=conf)


Alternatively pass the jars to your pyspark job via the --jars flag:

.. code-block:: sh

    $ spark-submit --jars `bin/sagemakerpyspark-jars`

If you want to play around in interactive mode, the pyspark shell can be used too:

.. code-block:: sh

    $ pyspark --jars `bin/sagemakerpyspark-jars`

You can also use the --packages flag and pass in the Maven coordinates for SageMaker Spark:

.. code-block:: sh

    $ pyspark --packages com.amazonaws:sagemaker-spark_2.11:spark_2.1.1-1.0



Training and Hosting a K-Means Clustering model using SageMaker PySpark
-----------------------------------------------------------------------

A KMeansSageMakerEstimator runs a training job using the Amazon SageMaker KMeans algorithm upon
invocation of fit(), returning a SageMakerModel.

.. code-block:: python

    from sagemaker_pyspark import IAMRole
    from sagemaker_pyspark.algorithms import KMeansSageMakerEstimator

    iam_role = "arn:aws:iam:0123456789012:role/MySageMakerRole"

    region = "us-east-1"
    training_data = spark.read.format("libsvm").option("numFeatures", "784")
      .load("s3a://sagemaker-sample-data-{}/spark/mnist/train/".format(region))

    test_data = spark.read.format("libsvm").option("numFeatures", "784")
      .load("s3a://sagemaker-sample-data-{}/spark/mnist/train/".format(region))

    kmeans_estimator = KMeansSageMakerEstimator(
        trainingInstanceType="ml.m4.xlarge",
        trainingInstanceCount=1,
        endpointInstanceType="ml.m4.xlarge",
        endpointInitialInstanceCount=1,
        sagemakerRole=IAMRole(iam_role))

    kmeans_estimator.setK(10)
    kmeans_estimator.setFeatureDim(784)

    kmeans_model = kmeans_estimator.fit(training_data)

    transformed_data = kmeans_model.transform(test_data)
    transformed_data.show()


The SageMakerEstimator expects an input DataFrame with a column named "features" that holds a
Spark ML  Vector. The estimator also serializes a "label" column of Doubles if present. Other
columns are ignored. The dimension of this input vector should be equal to the feature dimension
given as a hyperparameter.

The Amazon SageMaker KMeans algorithm accepts many parameters, but K (the number of clusters) and
FeatureDim (the number of features per Row) are required.

You can set other hyperparameters, See the docs (link), or run

.. code-block:: python

    kmeans_estimator.explainParams()

After training is complete, an Amazon SageMaker Endpoint is created to host the model and serve
predictions. Upon invocation of transform(), the SageMakerModel predicts against their hosted
model. Like the SageMakerEstimator, the SageMakerModel expects an input DataFrame with a column
named "features" that holds a Spark ML Vector equal in dimension to the value of the FeatureDim
parameter.


API Reference
-------------

If you are looking for information on a specific class or method, this is where its found.

.. toctree::
    :maxdepth: 2

    api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
