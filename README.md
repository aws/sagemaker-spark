# <img alt="SageMaker" src="branding/icon/sagemaker-banner.png" height="100">

# SageMaker Spark

SageMaker Spark is an open source Spark library for [Amazon SageMaker](https://aws.amazon.com/sagemaker/). With SageMaker Spark you construct Spark ML `Pipeline`s using Amazon SageMaker stages. These pipelines interleave native Spark ML stages and stages that interact with SageMaker training and model hosting.

With SageMaker Spark, you can train on Amazon SageMaker from Spark `DataFrame`s using **Amazon-provided ML algorithms**
like K-Means clustering or XGBoost, and make predictions on `DataFrame`s against
SageMaker endpoints hosting your trained models, and, if you have **your own ML algorithms** built
into SageMaker compatible Docker containers, you can use SageMaker Spark to train and infer on `DataFrame`s with your
own algorithms -- **all at Spark scale.**

## Table of Contents
* [Getting SageMaker Spark](#getting-sagemaker-spark)
  * [Scala](#scala)
* [Running SageMaker Spark](#running-sagemaker-spark)
  * [Running SageMaker Spark Applications with spark-shell or <code>spark-submit</code>](#running-sagemaker-spark-applications-with-spark-shell-or-spark-submit)
  * [Running SageMaker Spark Applications on EMR](#running-sagemaker-spark-applications-on-emr)
  * [Python](#python)
  * [S3 FileSystem Schemes](#s3-filesystem-schemes)
  * [API Documentation](#api-documentation)
* [Getting Started: K-Means Clustering on SageMaker with SageMaker Spark SDK](#getting-started-k-means-clustering-on-sagemaker-with-sagemaker-spark-sdk)
* [Example: Using SageMaker Spark with Any SageMaker Algorithm](#example-using-sagemaker-spark-with-any-sagemaker-algorithm)
* [Example: Using SageMakerEstimator and SageMakerModel in a Spark Pipeline](#example-using-sagemakerestimator-and-sagemakermodel-in-a-spark-pipeline)
* [Example: Using Multiple SageMakerEstimators and SageMakerModels in a Spark Pipeline](#example-using-multiple-sagemakerestimators-and-sagemakermodels-in-a-spark-pipeline)
* [Example: Creating a SageMakerModel](#example-creating-a-sagemakermodel)
  * [SageMakerModel From an Endpoint](#sagemakermodel-from-an-endpoint)
  * [SageMakerModel From Model Data in S3](#sagemakermodel-from-model-data-in-s3)
  * [SageMakerModel From a Previously Completed Training Job](#sagemakermodel-from-a-previously-completed-training-job)
* [Example: Tearing Down Amazon SageMaker Endpoints](#example-tearing-down-amazon-sagemaker-endpoints)
* [Configuring an IAM Role](#configuring-an-iam-role)
* [SageMaker Spark: In-Depth](#sagemaker-spark-in-depth)
  * [The Amazon Record format](#the-amazon-record-format)
  * [Serializing and Deserializing for Inference](#serializing-and-deserializing-for-inference)
* [License](#license)

## Getting SageMaker Spark

### Scala

SageMaker Spark for Scala is available in the Maven central repository:

```
<dependency>
    <groupId>com.amazonaws</groupId>
    <artifactId>sagemaker-spark_2.11</artifactId>
    <version>spark_2.2.0-1.0</version>
</dependency>
```

Or, if your project depends on Spark 2.1:

```
<dependency>
    <groupId>com.amazonaws</groupId>
    <artifactId>sagemaker-spark_2.11</artifactId>
    <version>spark_2.1.1-1.0</version>
</dependency>
```

You can also build SageMaker Spark from source. See [sagemaker-spark-sdk](sagemaker-spark-sdk) for more on
building SageMaker Spark from source.

### Python

See the [sagemaker-pyspark-sdk](sagemaker-pyspark-sdk) for more on installing and running SageMaker PySpark.

## Running SageMaker Spark

SageMaker Spark depends on hadoop-aws-2.8.1. To run Spark applications that depend on SageMaker Spark, you need to
build Spark with Hadoop 2.8. However, if you are running Spark applications on EMR, you can use Spark built with Hadoop 2.7.

Apache Spark currently distributes binaries built against Hadoop-2.7, but not 2.8.
See the [Spark documentation](https://spark.apache.org/docs/2.2.0/hadoop-provided.html) for more on building Spark
with Hadoop 2.8.

SageMaker Spark needs to be added to both the driver and executor classpaths.

### Running SageMaker Spark Applications with `spark-shell` or `spark-submit`

You can submit SageMaker Spark and the AWS Java Client as dependencies with the "--jars" flag, or take a dependency
on SageMaker Spark in Maven using the "--package" flag.

1. Install Hadoop-2.8. [https://hadoop.apache.org/docs/r2.8.0/](https://hadoop.apache.org/docs/r2.8.0/)
2. Build Spark 2.2 with Hadoop-2.8. The [Spark documentation](https://spark.apache.org/docs/2.2.0/hadoop-provided.html)
has guidance on building Spark with your own Hadoop installation.
3. Run ```spark-shell``` or ```spark-submit``` with the `--packages` flag:

```
spark-shell --packages com.amazonaws:sagemaker-spark_2.11:spark_2.2.0-1.0
```

### Running SageMaker Spark Applications on EMR

You can run SageMaker Spark applications on an EMR cluster just like any other Spark application by
submitting your Spark application jar and the SageMaker Spark dependency jars with the --jars or --packages flags.

SageMaker Spark is pre-installed on EMR releases since 5.11.0. You can run your SageMaker Spark application
on EMR by submitting your Spark application jar and any additional dependencies your Spark application uses.

SageMaker Spark applications have also been verified to be compatible with EMR-5.6.0 (which runs Spark 2.1) and EMR-5-8.0
(which runs Spark 2.2). When submitting your Spark application to an earlier EMR release, use the `--packages` flag to
depend on a recent version of the AWS Java SDK:  

```
spark-submit
  --packages com.amazonaws:aws-java-sdk:1.11.238 \
  --deploy-mode cluster \
  --conf spark.driver.userClassPathFirst=true \
  --conf spark.executor.userClassPathFirst=true \
  --jars SageMakerSparkApplicationJar.jar,...
  ...
```

The `spark.driver.userClassPathFirst=true` and `spark.executor.userClassPathFirst=true` properties are required so that
the Spark cluster will use the AWS Java SDK dependencies with SageMaker, rather than the AWS Java SDK installed on these
earlier EMR clusters.

For more on running Spark application on EMR, see the
[EMR Documentation](http://docs.aws.amazon.com/emr/latest/ReleaseGuide/emr-spark-submit-step.html) on submitting a step.

### Python

See the [sagemaker-pyspark-sdk](sagemaker-pyspark-sdk) for more on installing and running SageMaker PySpark.

### S3 FileSystem Schemes

EMR allows you to read and write data using the EMR FileSystem (EMRFS), accessed through Spark with "s3://":

```scala
spark.read.format("libsvm").load("s3://my-bucket/my-prefix")
```

In other execution environments, you can use the S3A schema to use the S3A FileSystem "s3a://" to read and write data:

```scala
spark.read.format("libsvm").load("s3a://my-bucket/my-prefix")
```

In the code examples in this README, we use "s3://" to use the [EMRFS](http://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-fs.html),
or "s3a://" to use the [S3A system](https://wiki.apache.org/hadoop/AmazonS3), which is recommended over "s3n://".

### API Documentation

You can view the [Scala API Documentation for SageMaker Spark here.](https://aws.github.io/sagemaker-spark/)

You can view the [PySpark API Documentation for SageMaker Spark here.](http://sagemaker-pyspark.readthedocs.io/en/latest/)

## Getting Started: K-Means Clustering on SageMaker with SageMaker Spark SDK
 
This example walks through using SageMaker Spark to train on a Spark DataFrame using a SageMaker-provided algorithm,
host the resulting model on SageMaker Spark, and making predictions on a Spark DataFrame using that hosted model.

We'll cluster handwritten digits in the MNIST dataset, which we've made available in LibSVM format at 
`s3://sagemaker-sample-data-us-east-1/spark/mnist/train/mnist_train.libsvm`.

You can start a Spark shell with SageMaker Spark

```
spark-shell --packages com.amazonaws:sagemaker-spark_2.11:spark_2.1.1-1.0
```

1. Create your Spark Session and load your training and test data into DataFrames:
```scala
val spark = SparkSession.builder.getOrCreate

// load mnist data as a dataframe from libsvm. replace this region with your own.
val region = "us-east-1"
val trainingData = spark.read.format("libsvm")
  .option("numFeatures", "784")
  .load(s"s3://sagemaker-sample-data-$region/spark/mnist/train/")

val testData = spark.read.format("libsvm")
  .option("numFeatures", "784")
  .load(s"s3://sagemaker-sample-data-$region/spark/mnist/test/")
```

The `DataFrame` consists of a column named "label" of Doubles, indicating the digit for each example,
and a column named "features" of Vectors:

```scala
trainingData.show

+-----+--------------------+
|label|            features|
+-----+--------------------+
|  5.0|(784,[152,153,154...|
|  0.0|(784,[127,128,129...|
|  4.0|(784,[160,161,162...|
|  1.0|(784,[158,159,160...|
|  9.0|(784,[208,209,210...|
|  2.0|(784,[155,156,157...|
|  1.0|(784,[124,125,126...|
|  3.0|(784,[151,152,153...|
|  1.0|(784,[152,153,154...|
|  4.0|(784,[134,135,161...|
|  3.0|(784,[123,124,125...|
|  5.0|(784,[216,217,218...|
|  3.0|(784,[143,144,145...|
|  6.0|(784,[72,73,74,99...|
|  1.0|(784,[151,152,153...|
|  7.0|(784,[211,212,213...|
|  2.0|(784,[151,152,153...|
|  8.0|(784,[159,160,161...|
|  6.0|(784,[100,101,102...|
|  9.0|(784,[209,210,211...|
+-----+--------------------+
```

2. Construct a `KMeansSageMakerEstimator`, which extends `SageMakerEstimator`, which is a Spark `Estimator`.
You need to pass in an Amazon SageMaker-compatible
IAM Role that Amazon SageMaker will use to make AWS service calls on your behalf (or configure SageMaker Spark
to [get this from Spark Config](#configuring-iam-role-and-s3-buckets)). Consult the API Documentation for a
complete list of parameters.

In this example, we are setting the "k" and "feature_dim" hyperparameters, corresponding to the number
of clusters we want and to the number of dimensions in our training dataset, respectively.

```scala

// Replace this IAM Role ARN with your own.
val roleArn = "arn:aws:iam::account-id:role/rolename"

val estimator = new KMeansSageMakerEstimator(
  sagemakerRole = IAMRole(roleArn),
  trainingInstanceType = "ml.p2.xlarge",
  trainingInstanceCount = 1,
  endpointInstanceType = "ml.c4.xlarge",
  endpointInitialInstanceCount = 1)
  .setK(10).setFeatureDim(784)
```

3. To train and host your model, call `fit()` on your training `DataFrame`:

```scala
val model = estimator.fit(trainingData)
```

What happens in this call to `fit()`?

1. SageMaker Spark serializes your `DataFrame` and uploads the
serialized training data to S3. For the K-Means algorithm, SageMaker Spark converts the `DataFrame` to the [Amazon Record
format](#the-amazon-record-format).
SageMaker Spark will create an S3 bucket for you that your IAM role can access if you do not provide an S3 Bucket in
the constructor.
2. SageMaker Spark sends a `CreateTrainingJobRequest` to Amazon SageMaker to run a Training Job with one `p2.xlarge` on the data in S3, configured with the
values you pass in to the `SageMakerEstimator`, and polls for completion of the Training Job.
In this example, we are sending a CreateTrainingJob request to run a k-means clustering Training Job on Amazon SageMaker
on serialized data we uploaded from your `DataFrame`. When training completes, the Amazon SageMaker service puts
a serialized model in an S3 bucket you own (or the default bucket created by SageMaker Spark).
3. After training completes, SageMaker Spark sends a `CreateModelRequest`, a `CreateEndpointConfigRequest`, and a
`CreateEndpointRequest` and polls for completion, each configured with the values you pass in to the SageMakerEstimator.
This Endpoint will initially be backed by one `c4.xlarge`.

4. To make inferences using the Endpoint hosting our model, call `transform()` on the `SageMakerModel` returned by `fit()`.

```scala
val transformedData = model.transform(testData)
transformedData.show
+-----+--------------------+-------------------+---------------+
|label|            features|distance_to_cluster|closest_cluster|
+-----+--------------------+-------------------+---------------+
|  5.0|(784,[152,153,154...|  1767.897705078125|            4.0|
|  0.0|(784,[127,128,129...|  1392.157470703125|            5.0|
|  4.0|(784,[160,161,162...| 1671.5711669921875|            9.0|
|  1.0|(784,[158,159,160...| 1182.6082763671875|            6.0|
|  9.0|(784,[208,209,210...| 1390.4002685546875|            0.0|
|  2.0|(784,[155,156,157...|  1713.988037109375|            1.0|
|  1.0|(784,[124,125,126...| 1246.3016357421875|            2.0|
|  3.0|(784,[151,152,153...|  1753.229248046875|            4.0|
|  1.0|(784,[152,153,154...|  978.8394165039062|            2.0|
|  4.0|(784,[134,135,161...|  1623.176513671875|            3.0|
|  3.0|(784,[123,124,125...|  1533.863525390625|            4.0|
|  5.0|(784,[216,217,218...|  1469.357177734375|            6.0|
|  3.0|(784,[143,144,145...|  1736.765869140625|            4.0|
|  6.0|(784,[72,73,74,99...|   1473.69384765625|            8.0|
|  1.0|(784,[151,152,153...|    944.88720703125|            2.0|
|  7.0|(784,[211,212,213...| 1285.9071044921875|            3.0|
|  2.0|(784,[151,152,153...| 1635.0125732421875|            1.0|
|  8.0|(784,[159,160,161...| 1436.3162841796875|            6.0|
|  6.0|(784,[100,101,102...| 1499.7366943359375|            7.0|
|  9.0|(784,[209,210,211...| 1364.6319580078125|            6.0|
+-----+--------------------+-------------------+---------------+

```

In this call to `transform()`, the `SageMakerModel` serializes chunks of the input `DataFrame` and sends them to the
Endpoint using the SageMakerRuntime `InvokeEndpoint` API. The `SageMakerModel` deserializes the Endpoint's responses,
which contain predictions, and appends the prediction columns to the input `DataFrame`.

## Example: Using SageMaker Spark with Any SageMaker Algorithm

The `SageMakerEstimator` is an `org.apache.spark.ml.Estimator` that trains a model on Amazon SageMaker.

SageMaker Spark provides several classes that extend `SageMakerEstimator` to run particular algorithms, like `KMeansSageMakerEstimator`
to run the SageMaker-provided k-means algorithm, or `XGBoostSageMakerEstimator` to run the SageMaker-provided XGBoost
algorithm. These classes are just `SageMakerEstimator`s with certain default values passed in. You can use SageMaker Spark with
any algorithm that runs on Amazon SageMaker by creating a SageMakerEstimator.

Instead of creating a KMeansSageMakerEstimator, you can create an equivalent SageMakerEstimator:

```scala
val estimator = new SageMakerEstimator(
  trainingImage =
    "382416733822.dkr.ecr.us-east-1.amazonaws.com/kmeans:1",
  modelImage =
    "382416733822.dkr.ecr.us-east-1.amazonaws.com/kmeans:1",
  requestRowSerializer = new ProtobufRequestRowSerializer(),
  responseRowDeserializer = new KMeansProtobufResponseRowDeserializer(),
  hyperParameters = Map("k" -> "10", "feature_dim" -> "784"),
  sagemakerRole = IAMRole(roleArn),
  trainingInstanceType = "ml.p2.xlarge",
  trainingInstanceCount = 1,
  endpointInstanceType = "ml.c4.xlarge",
  endpointInitialInstanceCount = 1,
  trainingSparkDataFormat = "sagemaker")
```

* `trainingImage` identifies the Docker registry path to the training image containing your custom code. In this case,
this points to the us-east-1 k-means image.
* `modelImage` identifies the Docker registry path to the image containing inference code. Amazon SageMaker k-means 
uses the same image to train and to host trained models.
* `requestRowSerializer` implements `com.amazonaws.services.sagemaker.sparksdk.transformation.RequestRowSerializer`.
A `RequestRowSerializer` serializes `org.apache.spark.sql.Row`s in the input `DataFrame` to send them to the model hosted in Amazon SageMaker for inference.
This is passed to the SageMakerModel returned by `fit`. In this case, we pass in a `RequestRowSerializer` that serializes
`Row`s to the Amazon Record protobuf format. See [Serializing and Deserializing for Inference](#serializing-and-deserializing-for-inference)
for more information on how SageMaker Spark makes inferences. 
* `responseRowDeserializer` Implements
`com.amazonaws.services.sagemaker.sparksdk.transformation.ResponseRowDeserializer`. A `ResponseRowDeserializer` deserializes
responses containing predictions from the Endpoint back into columns in a `DataFrame`.
* `hyperParameters` is a `Map[String, String]` that the `trainingImage` will use to set training hyperparameters.
* `trainingSparkDataFormat` specifies the data format that Spark uses when uploading training data from a `DataFrame`
to S3.

SageMaker Spark needs the trainingSparkDataFormat to tell Spark how to write the DataFrame to S3 for the `trainingImage` to
train on. In this example, "sagemaker" tells Spark to write the data as
RecordIO-encoded [Amazon Records](#the-amazon-record-format), but your own algorithm may take another data format.
You can pass in any format that Spark supports as long as your `trainingImage` can train using that data format,
such as "csv", "parquet", "com.databricks.spark.csv", or "libsvm."

SageMaker Spark also needs a `RequestRowSerializer` to serialize Spark `Row`s to a
data format the `modelImage` can deserialize, and a `ResponseRowDeserializer` to deserialize responses that contain
predictions from the `modelImage` back into Spark `Row`s. See [Serializing and Deserializing for Inference](#serializing-and-deserializing-for-inference)
for more details.

## Example: Using SageMakerEstimator and SageMakerModel in a Spark Pipeline

`SageMakerEstimator`s and `SageMakerModel`s can be used in `Pipeline`s. In this
example, we run `org.apache.spark.ml.feature.PCA` on our Spark cluster, then train and infer using Amazon SageMaker's
K-Means on the output column from `PCA`:

```scala
val pcaEstimator = new PCA()
  .setInputCol("features")
  .setOutputCol("projectedFeatures")
  .setK(50)

val kMeansSageMakerEstimator = new KMeansSageMakerEstimator(
  sagemakerRole = IAMRole(roleArn),
  requestRowSerializer =
    new ProtobufRequestRowSerializer(featuresColumnName = "projectedFeatures"),
  trainingSparkDataFormatOptions = Map("featuresColumnName" -> "projectedFeatures"),
  trainingInstanceType = "ml.p2.xlarge",
  trainingInstanceCount = 1,
  endpointInstanceType = "ml.c4.xlarge",
  endpointInitialInstanceCount = 1)
  .setK(10).setFeatureDim(50)

val pipeline = new Pipeline().setStages(Array(pcaEstimator, kMeansSageMakerEstimator))

// train
val pipelineModel = pipeline.fit(trainingData)

val transformedData = pipelineModel.transform(testData)
transformedData.show()

+-----+--------------------+--------------------+-------------------+---------------+
|label|            features|   projectedFeatures|distance_to_cluster|closest_cluster|
+-----+--------------------+--------------------+-------------------+---------------+
|  5.0|(784,[152,153,154...|[880.731433034386...|     1500.470703125|            0.0|
|  0.0|(784,[127,128,129...|[1768.51722024166...|      1142.18359375|            4.0|
|  4.0|(784,[160,161,162...|[704.949236329314...|  1386.246826171875|            9.0|
|  1.0|(784,[158,159,160...|[-42.328192193771...| 1277.0736083984375|            5.0|
|  9.0|(784,[208,209,210...|[374.043902028333...|   1211.00927734375|            3.0|
|  2.0|(784,[155,156,157...|[941.267714528850...|  1496.157958984375|            8.0|
|  1.0|(784,[124,125,126...|[30.2848596410594...| 1327.6766357421875|            5.0|
|  3.0|(784,[151,152,153...|[1270.14374062052...| 1570.7674560546875|            0.0|
|  1.0|(784,[152,153,154...|[-112.10792566485...|     1037.568359375|            5.0|
|  4.0|(784,[134,135,161...|[452.068280676606...| 1165.1236572265625|            3.0|
|  3.0|(784,[123,124,125...|[610.596447285397...|  1325.953369140625|            7.0|
|  5.0|(784,[216,217,218...|[142.959601818422...| 1353.4930419921875|            5.0|
|  3.0|(784,[143,144,145...|[1036.71862533658...| 1460.4315185546875|            7.0|
|  6.0|(784,[72,73,74,99...|[996.740157435754...| 1159.8631591796875|            2.0|
|  1.0|(784,[151,152,153...|[-107.26076167417...|   960.963623046875|            5.0|
|  7.0|(784,[211,212,213...|[619.771820430940...|   1245.13623046875|            6.0|
|  2.0|(784,[151,152,153...|[850.152101817161...|  1304.437744140625|            8.0|
|  8.0|(784,[159,160,161...|[370.041887230547...| 1192.4781494140625|            0.0|
|  6.0|(784,[100,101,102...|[546.674328209335...|    1277.0908203125|            2.0|
|  9.0|(784,[209,210,211...|[-29.259112927426...| 1245.8182373046875|            6.0|
+-----+--------------------+--------------------+-------------------+---------------+
```

* `requestRowSerializer =
      new ProtobufRequestRowSerializer(featuresColumnName = "projectedFeatures")` tells the `SageMakerModel` returned
      by `fit()` to infer on the features in the "projectedFeatures" column
* `trainingSparkDataFormatOptions = Map("featuresColumnName" -> "projectedFeatures")` tells the `SageMakerProtobufWriter`
 that Spark is using to write the `DataFrame` as format "sagemaker" to serialize the "projectedFeatures" column when
 writing Amazon Records for training.


## Example: Using Multiple SageMakerEstimators and SageMakerModels in a Spark Pipeline

We can use multiple `SageMakerEstimator`s and `SageMakerModel`s in a pipeline. Here, we use
SageMaker's PCA algorithm to reduce a dataset with 50 dimensions to a dataset with 20 dimensions, then
use SageMaker's K-Means algorithm to train on the 20-dimension data.

```scala
val pcaEstimator = new PCASageMakerEstimator(sagemakerRole = IAMRole(sagemakerRole),
  trainingInstanceType = "ml.p2.xlarge",
  trainingInstanceCount = 1,
  endpointInstanceType = "ml.c4.xlarge",
  endpointInitialInstanceCount = 1
  responseRowDeserializer = new PCAProtobufResponseRowDeserializer(
    projectionColumnName = "projectionDim20"),
  trainingInputS3DataPath = S3DataPath(trainingBucket, inputPrefix),
  trainingOutputS3DataPath = S3DataPath(trainingBucket, outputPrefix),
  endpointCreationPolicy = EndpointCreationPolicy.CREATE_ON_TRANSFORM)
  .setNumComponents(20).setFeatureDim(50)

val kmeansEstimator = new KMeansSageMakerEstimator(sagemakerRole = IAMRole(sagemakerRole),
  trainingInstanceType = "ml.p2.xlarge",
  trainingInstanceCount = 1,
  endpointInstanceType = "ml.c4.xlarge",
  endpointInitialInstanceCount = 1
  trainingSparkDataFormatOptions = Map("featuresColumnName" -> "projectionDim20"),
  requestRowSerializer = new ProtobufRequestRowSerializer(
    featuresColumnName = "projectionDim20"),
  responseRowDeserializer = new KMeansProtobufResponseRowDeserializer(),
  trainingInputS3DataPath = S3DataPath(trainingBucket, inputPrefix),
  trainingOutputS3DataPath = S3DataPath(trainingBucket, outputPrefix),
  endpointCreationPolicy = EndpointCreationPolicy.CREATE_ON_TRANSFORM)
  .setK(10).setFeatureDim(20)

val pipeline = new Pipeline().setStages(Array(pcaEstimator, kmeansEstimator))

val model = pipeline.fit(dataset)

// For expediency, transforming the training dataset:
val transformedData = model.transform(dataset)
transformedData.show()

+-----+--------------------+--------------------+-------------------+---------------+
|label|            features|     projectionDim20|distance_to_cluster|closest_cluster|
+-----+--------------------+--------------------+-------------------+---------------+
|  1.0|[-0.7927307,-11.2...|[5.50362682342529...|  45.03189468383789|            1.0|
|  1.0|[-3.762671,-5.853...|[-2.1558122634887...|  41.79889678955078|            1.0|
|  1.0|[-2.0988898,-2.40...|[4.53881502151489...| 50.824703216552734|            1.0|
|  1.0|[-2.81075,-3.6481...|[0.97894239425659...|  52.78211975097656|            1.0|
|  1.0|[-2.14356,-4.0369...|[2.25758934020996...|  48.99141311645508|            1.0|
|  1.0|[-5.3773708,-15.3...|[-3.2523036003112...|  21.99374771118164|            1.0|
|  1.0|[-1.0369565,-16.5...|[-17.643878936767...| 29.127044677734375|            3.0|
|  1.0|[-2.019725,-3.226...|[1.41068196296691...|   51.7830696105957|            1.0|
|  1.0|[-4.3821997,-0.98...|[-0.8335087299346...| 53.921058654785156|            1.0|
|  1.0|[-7.075208,-34.31...|[11.4329795837402...|  35.12031173706055|            3.0|
|  1.0|[-3.90454,-4.8401...|[-1.4304646253585...|  50.00594711303711|            1.0|
|  1.0|[0.9607103,-13.50...|[1.13785743713378...|  28.71956443786621|            1.0|
|  1.0|[-4.5025017,-15.2...|[2.66747045516967...| 25.419822692871094|            1.0|
|  1.0|[0.041773,-27.148...|[7.58121681213378...| 30.303693771362305|            3.0|
|  1.0|[-10.1477266,-39....|[-12.086886405944...|   35.9030647277832|            2.0|
|  1.0|[-3.09143,-6.4892...|[1.79180252552032...|  39.34271240234375|            1.0|
|  1.0|[-13.5285917,-32....|[7.62783145904541...| 35.040035247802734|            2.0|
|  1.0|[-4.189806,-16.04...|[1.41141772270202...| 25.123626708984375|            1.0|
|  1.0|[-12.77831508,-62...|[0.11281073093414...|  63.91242599487305|            2.0|
|  1.0|[-9.3934507,-12.5...|[-9.4945802688598...| 20.913305282592773|            1.0|
+-----+--------------------+--------------------+-------------------+---------------+

```
* `responseRowDeserializer = new PCAProtobufResponseRowDeserializer(
projectionColumnName = "projectionDim20")` tells the `SageMakerModel` attached to the PCA endpoint to deserialize
responses (which contain the lower-dimensional projections of the features vectors) into the column named "projectionDim20"
* `endpointCreationPolicy = EndpointCreationPolicy.CREATE_ON_TRANSFORM` tells the `SageMakerEstimator` to delay SageMaker
 Endpoint creation until it is needed to transform a `DataFrame`. 
* `trainingSparkDataFormatOptions = Map("featuresColumnName" -> "projectionDim20"),
   requestRowSerializer = new ProtobufRequestRowSerializer(
       featuresColumnName = "projectionDim20")` these lines tell the `KMeansSageMakerEstimator`
       to respectively train and infer on the features in the "projectionDim20" column.

## Example: Creating a SageMakerModel

SageMaker Spark supports attaching `SageMakerModel`s to an existing SageMaker endpoint, or to an Endpoint created by
reference to model data in S3, or to a previously completed Training Job.

This allows you to use SageMaker Spark just for model hosting and inference on Spark-scale `DataFrame`s without running
a new Training Job.

### SageMakerModel From an Endpoint

You can attach a `SageMakerModel` to an endpoint that has already been created. Supposing an endpoint with name
"my-endpoint-name" is already in service and hosting a SageMaker K-Means model:

```scala
val model = SageMakerModel
  .fromEndpoint(endpointName = "my-endpoint-name",
                requestRowSerializer = new ProtobufRequestRowSerializer(
                  featuresColumnName = "MyFeaturesColumn"),
                responseRowDeserializer = new KMeansProtobufResponseRowDeserializer(
                  distanceToClusterColumnName = "DistanceToCluster",
                  closestClusterColumnName = "ClusterLabel"
                ))
```

This `SageMakerModel` will, upon a call to `transform()`, serialize the column named
"MyFeaturesColumn" for inference, and append the columns "DistanceToCluster" and "ClusterLabel" to the `DataFrame`.

### SageMakerModel From Model Data in S3

You can create a SageMakerModel and an Endpoint by referring directly to your model data in S3:

```scala
val model = SageMakerModel
  .fromModelS3Path(modelPath = "s3://my-model-bucket/my-model-data/model.tar.gz",
                   modelExecutionRoleARN = "arn:aws:iam::account-id:role/rolename"
                   modelImage = 382416733822.dkr.ecr.us-east-1.amazonaws.com/kmeans:1",
                   endpointInstanceType = "ml.c4.xlarge",
                   endpointInitialInstanceCount = 1
                   requestRowSerializer = new ProtobufRequestRowSerializer(),
                   responseRowDeserializer = new KMeansProtobufResponseRowDeserializer()
                  )
```

### SageMakerModel From a Previously Completed Training Job

You can create a SageMakerModel and an Endpoint by referring to a previously-completed training job:

```scala
val model = SageMakerModel
  .fromTrainingJob(trainingJobName = "my-training-job-name",
                   modelExecutionRoleARN = "arn:aws:iam::account-id:role/rolename"
                   modelImage = 382416733822.dkr.ecr.us-east-1.amazonaws.com/kmeans:1",
                   endpointInstanceType = "ml.c4.xlarge",
                   endpointInitialInstanceCount = 1
                   requestRowSerializer = new ProtobufRequestRowSerializer(),
                   responseRowDeserializer = new KMeansProtobufResponseRowDeserializer()
                  )

```

## Example: Tearing Down Amazon SageMaker Endpoints

SageMaker Spark provides a utility for deleting Endpoints created by a SageMakerModel:

```scala
val cleanup = new SageMakerResourceCleanup(sagemakerClient)
cleanup.deleteResources(model.getCreatedResources)

```

## Configuring an IAM Role

SageMaker Spark allows you to add your IAM Role ARN to your Spark Config so that you don't have to keep passing in
`IAMRole("arn:aws:iam::account-id:role/rolename")`.

Add an entry to your Spark Config with key `com.amazonaws.services.sagemaker.sparksdk.sagemakerrole` whose value is your
Amazon SageMaker-compatible IAM Role. `SageMakerEstimator` will look for this role if it is not supplied in the constructor.

## SageMaker Spark: In-Depth

### The Amazon Record format

`KMeansSageMakerEstimator`, `PCASageMakerEstimator`, and `LinearLearnerSageMakerEstimator` all serialize `DataFrame`s
to the Amazon Record protobuf format with each Record encoded in
[RecordIO](https://mxnet.incubator.apache.org/architecture/note_data_loading.html).
They do this by passing in "sagemaker" to the `trainingSparkDataFormat` constructor argument, which configures Spark
to use the `SageMakerProtobufWriter` to serialize Spark `DataFrame`s.

Writing a `DataFrame` using the "sagemaker"
format serializes a column named "label", expected to contain
`Double`s, and a column named "features", expected to contain a Sparse or Dense `org.apache.mllib.linalg.Vector`.
If the features column contains a `SparseVector`, SageMaker Spark sparsely-encodes the `Vector` into the Amazon Record.
If the features column contains a `DenseVector`, SageMaker Spark densely-encodes the `Vector` into the Amazon Record.

You can choose which columns the `SageMakerEstimator` chooses as its "label" and "features" columns by passing in 
a `trainingSparkDataFormatOptions` `Map[String, String]` with keys "labelColumnName" and "featuresColumnName" and with
values corresponding to the names of your chosen label and features columns.

You can also write Amazon Records using SageMaker Spark by using the "sagemaker" format directly:

```scala
myDataFrame.write
    .format("sagemaker")
    .option("labelColumnName", "myLabelColumn")
    .option("featuresColumnName", "myFeaturesColumn")
    .save("s3://my-s3-bucket/my-s3-prefix")
```

By default, `SageMakerEstimator` deletes the RecordIO-encoded Amazon Records in S3 following training on Amazon 
SageMaker. You can choose to allow the data to persist in S3 by passing in `deleteStagingDataAfterTraining = true` to 
`SageMakerEstimator`.

See the [AWS Documentation on Amazon Records](https://aws.amazon.com/sagemaker/latest/dg/cdf-training.html) for
more information on Amazon Records.

### Serializing and Deserializing for Inference

`SageMakerEstimator.fit()` returns a `SageMakerModel`, which transforms a `DataFrame` by calling `InvokeEndpoint` on
an Amazon SageMaker Endpoint. `InvokeEndpointRequest`s carry serialized `Row`s as their payload.`Row`s in the `DataFrame`
are serialized for predictions against an Endpoint using a `RequestRowSerializer`. Responses from an Endpoint containing
predictions are deserialized into Spark `Row`s and appended as columns in a `DataFrame` using a `ResponseRowDeserializer.`

Internally, `SageMakerModel.transform` calls `mapPartitions` to distribute the work
of serializing Spark `Row`s, constructing and sending `InvokeEndpointRequest`s to an Endpoint, and deserializing
`InvokeEndpointResponse`s across a Spark cluster. Because each `InvokeEndpointRequest` can carry only 5MB, each 
Spark partition creates a
`com.amazonaws.services.sagemaker.sparksdk.transformation.util.RequestBatchIterator` to iterate over its partition,
sending prediction requests to the Endpoint in 5MB increments.

`RequestRowSerializer.serializeRow()` converts a `Row` to an `Array[Byte]`.
The `RequestBatchIterator` appends these byte arrays to
form the request body of an `InvokeEndpointRequest`.

For example, the
`com.amazonaws.services.sagemaker.sparksdk.transformation.ProtobufRequestRowSerializer` creates one
RecordIO-encoded Amazon Record per input row by serializing the "features" column in each row, and wrapping each
Amazon Record in the RecordIO header.

`ResponseRowDeserializer.deserializeResponse()` converts an `Array[Byte]` containing predictions from an Endpoint to 
an `Iterator[Row]`to appends columns containing these predictions to the `DataFrame` being transformed by the
`SageMakerModel`.

For comparison, SageMaker's XGBoost uses LibSVM-formatted data for inference (as well as training), and responds with a comma-delimited list of predictions.
Accordingly, SageMaker Spark uses `com.amazonaws.services.sagemaker.sparksdk.transformation.LibSVMRequestRowSerializer`
to serialize rows into LibSVM-formatted data, and uses `com.amazonaws.services.sagemaker.sparksdk.transformation.XGBoostCSVResponseRowDeserializer`
to deserialize the response into a column of predictions.

To support your own model image's data formats for inference, you can implement your own `RequestRowSerializer` and `ResponseRowDeserializer`.

## License

SageMaker Spark is licensed under [Apache-2.0](https://github.com/aws/sagemaker-spark/LICENSE.txt).
