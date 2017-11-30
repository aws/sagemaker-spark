# <img alt="SageMaker" src="../branding/icon/sagemaker-banner.png" height="100">

# SageMaker Spark for Scala

SageMaker Spark is an open source Spark library for Amazon SageMaker. With SageMaker Spark you construct Spark ML `Pipeline`s using Amazon SageMaker stages. These pipelines         interleave native Spark ML stages and stages that interact with SageMaker training and model hosting.

With SageMaker Spark, you can train on Amazon SageMaker from Spark `DataFrame`s using **Amazon-provided ML algorithms**
like K-Means clustering or XGBoost, and make predictions on `DataFrame`s against
SageMaker endpoints hosting your trained models, and, if you have **your own ML algorithms** built
into SageMaker compatible Docker containers, you can use SageMaker Spark to train and infer on `DataFrame`s with your
own algorithms -- **all at Spark scale.**

## Getting SageMaker Spark for Scala

### Maven

SageMaker Spark SDK for Scala is available in the Maven central repository. If your project is built with Maven,
add the following to your pom.xml file:

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
### SBT

If your project is built with sbt, add the following to your build.sbt file:

```
libraryDependencies += "com.amazonaws" % "sagemaker-spark_2.11" % "spark_2.2.0-1.0"

```

### Building from source

This package is built using [sbt](http://www.scala-sbt.org/). To run unit tests and build this package from source, run,
install sbt 1.x and run

```
sbt test; sbt package
```