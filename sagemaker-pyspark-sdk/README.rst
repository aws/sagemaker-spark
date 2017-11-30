.. image:: ../branding/icon/sagemaker-banner.png
    :height: 100px
    :alt: SageMaker

========================
Amazon SageMaker PySpark
========================

The SageMaker PySpark SDK provides a pyspark interface to Amazon SageMaker, allowing customers to
train using the Spark Estimator API, host their model on Amazon SageMaker, and make predictions
with their model using the Spark Transformer API. You can find the latest, most up to date,
documentation at `Read the Docs <http://sagemaker-pyspark.readthedocs.io>`_.

Table of Contents
-----------------

1. `Quick Start <#quick-start>`__
2. `Running on SageMaker Notebook Instances <#running-on-sagemaker-notebook-instances>`__
3. `Development <#development>`__

Quick Start
------------

sagemaker_pyspark works with python 2.7 and python 3.x. To install it use ``pip``:

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

    $ spark-submit --jars `sagemakerpyspark-jars` ...


If you want to play around in interactive mode, the pyspark shell can be used too:

.. code-block:: sh

    $ pyspark --jars `sagemakerpyspark-jars`



Training and Hosting a K-Means Clustering model using SageMaker PySpark
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A KMeansSageMakerEstimator runs a training job using the Amazon SageMaker KMeans algorithm upon
invocation of fit(), returning a SageMakerModel.

.. code-block:: python
    from sagemaker_pyspark import IAMRole
    from sagemaker_pyspark.algorithms import KMeansSageMakerEstimator

    iam_role = "arn:aws:iam:0123456789012:role/MySageMakerRole"

    training_data = spark.read.format("libsvm").option("numFeatures", "50") \
        .option("vectorType", "dense").load("s3a://some-bucket/some-data")

    kmeans_estimator = KMeansSageMakerEstimator(
        trainingInstanceType="ml.m4.xlarge",
        trainingInstanceCount=1,
        endpointInstanceType="ml.m4.xlarge",
        endpointInitialInstanceCount=1,
        sagemakerRole=IAMRole(iam_role))

    kmeans_estimator.setK(10)
    kmeans_estimator.setFeatureDim(50)

    kmeans_model = estimator.fit(training_data)

    transformed_data = kmeans_model.transform(training_data)
    transformed_data.show()

The SageMakerEstimator expects an input DataFrame with a column named "features" that holds a
Spark ML  Vector. The estimator also serializes a "label" column of Doubles if present. Other
columns are ignored. The dimension of this input vector should be equal to the feature dimension
given as a hyperparameter.

The Amazon SageMaker KMeans algorithm accepts many parameters, but K (the number of clusters) and
FeatureDim (the number of features per Row) are required.

You can set other hyperparameters, for details on them, run:

.. code-block:: python

    kmeans_estimator.explainParams()

After training is complete, an Amazon SageMaker Endpoint is created to host the model and serve
predictions. Upon invocation of transform(), the SageMakerModel predicts against their hosted
model. Like the SageMakerEstimator, the SageMakerModel expects an input DataFrame with a column
named "features" that holds a Spark ML Vector equal in dimension to the value of the FeatureDim
parameter.


Running on SageMaker Notebook Instances
---------------------------------------

sagemaker_pyspark comes pre-installed in the SageMaker Notebook Environment. There are 2 use
cases that we support:

- running on local spark
- connecting to an EMR spark cluster


Local Spark on SageMaker Notebook Instances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a notebook using the ``conda_python2`` or ``conda_python3`` Kernels. Then you can
initialize a spark context the same way it is described in the QuickStart section:

.. code-block:: python

    from pyspark import SparkContext, SparkConf
    import sagemaker_pyspark

    conf = (SparkConf()
            .set("spark.driver.extraClassPath", ":".join(sagemaker_pyspark.classpath_jars())))
    SparkContext(conf=conf)

Connecting to an EMR Spark Cluster
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note: Make sure your SageMaker Notebook instance can talk to your EMR Cluster. This means:

- They are in the same VPC or different `peered VPCs <http://docs.aws.amazon.com/AmazonVPC/latest/UserGuide/vpc-peering.html>`__.
- The EMR Cluster Security group allows TCP port 8998 on the SageMaker Notebook Security group to ingress.

Installing sagemaker_pyspark in a Spark EMR Cluster
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

sagemaker_pyspark works with ``EMR-5-8.0`` (which runs Spark 2.2). To install sagemaker_pyspark
in EMR:

Create a bootstrap script to install sagemaker_pyspark in your new EMR cluster:


.. code-block:: sh

    #!/bin/bash

    sudo pip install sagemaker_pyspark
    sudo /usr/bin/pip-3.4 install sagemaker_pyspark


Upload this script to an S3 bucket:

.. code-block:: sh

    $ aws s3 cp bootstrap.sh s3://your-bucket/prefix/

In the AWS Console launch a new EMR Spark Cluster,  set s3://your-bucket/prefix/bootstrap.sh  as the
bootstrap script. Make sure to:

- Run the Cluster in the same VPC as your SageMaker Notebook Instance.
- Provide an SSH Key that you have access to, as there will be some manual configuration required.

Once the cluster is launched, login to the master node:

.. code-block:: sh

    $ ssh -i /path/to/ssh-key.pem hadoop@your-emr-cluster-public-dns


Create a backup of the default spark configuration:

.. code-block:: sh

    $ cd /usr/lib/spark/conf
    $ sudo cp spark-defaults.conf spark-defaults.conf.bk

Grab the EMR classpath from the installed sagemaker_pyspark:

.. code-block:: sh

    $ sagemakerpyspark-emr-jars :

the output will be a ":" separated list of jar files. Copy the output and append it to the
``spark.driver.extraClassPath`` and ``spark.executor.extraClassPath`` sections of
``spark-defaults.conf``

Make sure that there is a ":" after the original classpath before you paste the sagemaker_pyspark
classpath.

Before proceeding to configure your Notebook instance, open port ``8998`` to allow ingress from the
security group in the Notebook instance.

Configure your SageMaker Notebook instance to connect to the cluster
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Open a terminal session in your notebook: new->terminal

Copy the default `sparkmagic config <https://github
.com/jupyter-incubator/sparkmagic/blob/master/sparkmagic/example_config.json>`__

You can download it in your terminal using:

.. code-block:: sh

    $ wget https://raw.githubusercontent
    .com/jupyter-incubator/sparkmagic/master/sparkmagic/example_config.json

In the `kernel_python_credentials` section, replace the `url` with
http://your-cluster-private-dns-name:8998`.

Override the default spark magic config

.. code-block:: sh

    $ cp example_config.json ~/.sparkmagic/config.json


Launch a notebook using either the ``pyspark2`` or ``pyspark3`` Kernel. As soon as you try to run
any code block, the notebook will connect to your spark cluster and get a ``SparkContext`` for you.


Development
-----------

Getting Started
~~~~~~~~~~~~~~~

Since sagemaker_pyspark depends on the Scala spark modules, you need to be able to build those.
Follow the instructions in `here <../sagemaker-spark-sdk/README.md>`__.

For the python side, assuming that you have python and ``virtualenv`` installed, set up your
environment and install the required dependencies like this instead of the
``pip install sagemaker_pyspark`` defined above:

.. code-block:: sh

    $ git clone https://github.com/aws/sagemaker-spark.git
    $ cd sagemaker-spark/sagemaker-pyspark-sdk/
    $ virtualenv venv
    ....
    $ . venv/bin/activate
    $ pip install -r requirements.txt
    $ pip install -e .

Running Tests
~~~~~~~~~~~~~

Our recommended way of running the tests is using pyenv + pyenv-virtualenv. This allows you to
test on different python versions, and to test the installed distribution instead of your local
files.

Install `pyenv <https://github.com/pyenv/pyenv>`__, `pyenv-virtualenv <https://github
.com/pyenv/pyenv-virtualenv>`__ and `pyenv-virtualenvwrapper <https://github
.com/pyenv/pyenv-virtualenvwrapper>`__

You can do this in OSX using `brew <https://brew.sh/>`__

.. code-block:: sh

    $ brew install pyenv pyenv-virtualenv pyenv-virtualenvwrapper

For linux you can just follow the steps in each of the package's Readme. Or if your distribution
has them as packages that is a good alternative.

make sure to add the pyenv and virtualenv init functions to your corresponding
shell init (**.bashrc**, **.zshrc**, etc):

.. code-block:: sh

    eval "$(pyenv init -)"
    eval "$(pyenv virtualenv-init -)"

Start a new shell once you do that to pick up your changes.

Setup the python version we need. At the moment we are testing with python
2.7, 3.5 and 3.6 so we need to install these versions:

.. code-block:: sh

    $ pyenv install 2.7.10
    $ pyenv install 3.5.2
    $ pyenv install 3.6.2

Set them as global versions

.. code-block:: sh

    $ pyenv global 2.7.10 3.5.2 3.6.2

Verify they show up when you do:

.. code-block:: sh

    $ pyenv versions

Restart your shell and run the command again to verify that it persists across shell sessions.

Now we just need to install tox to run our tests:

.. code-block:: sh

    $ pip install tox

Run the tests by running:

.. code-block:: sh

    $ tox

