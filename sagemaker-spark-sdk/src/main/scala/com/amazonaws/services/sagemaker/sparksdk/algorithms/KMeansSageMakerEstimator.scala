/*
 * Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * A copy of the License is located at
 *
 *   http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed
 * on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 * express or implied. See the License for the specific language governing
 * permissions and limitations under the License.
 */

package com.amazonaws.services.sagemaker.sparksdk.algorithms

import com.amazonaws.regions.DefaultAwsRegionProviderChain
import com.amazonaws.services.s3.{AmazonS3, AmazonS3ClientBuilder}
import com.amazonaws.services.securitytoken.{AWSSecurityTokenService, AWSSecurityTokenServiceClientBuilder}

import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.StructType

import com.amazonaws.services.sagemaker.{AmazonSageMaker, AmazonSageMakerClientBuilder}
import com.amazonaws.services.sagemaker.model.{S3DataDistribution, TrainingInputMode}
import com.amazonaws.services.sagemaker.sparksdk._
import com.amazonaws.services.sagemaker.sparksdk.EndpointCreationPolicy.EndpointCreationPolicy
import com.amazonaws.services.sagemaker.sparksdk.transformation.{RequestRowSerializer, ResponseRowDeserializer}
import com.amazonaws.services.sagemaker.sparksdk.transformation.deserializers.KMeansProtobufResponseRowDeserializer
import com.amazonaws.services.sagemaker.sparksdk.transformation.serializers.ProtobufRequestRowSerializer

/**
  * Common params for [[KMeansSageMakerEstimator]] with accessors
  */
private[algorithms] trait KMeansParams extends SageMakerAlgorithmParams {

  /**
    * The number of clusters to create (k). Must be > 1.
    */
  val k : IntParam = new IntParam(this, "k", "The number of clusters to create. " +
    "Must be > 1.", ParamValidators.gtEq(2))
  def getK: Int = $(k)

  /**
    * The initialization algorithm to choose centroids. Must be "random" or "kmeans++".
    * Default: "random".
    */
  val initMethod : Param[String] = new Param(this, "init_method",
    "The initialization algorithm to choose centroids. Supported options: 'random' and 'kmeans++'.",
    ParamValidators.inArray(Array("random", "kmeans++")))
  def getInitMethod: String = $(initMethod)

  /**
    * Maximum iterations for Lloyds EM procedure in the local kmeans used in finalized stage.
    * Must be > 0.
    * Default: 300.
    */
  val maxIter : IntParam = new IntParam(this, "local_lloyd_max_iter",
    "Maximum iterations for Lloyds EM procedure" +
    "in the local kmeans used in finalized stage. Must be > 0", ParamValidators.gtEq(1))
  def getMaxIter: Int = $(maxIter)

  /**
    * Tolerance for change in ssd for early stopping in local kmeans. Must be in range [0, 1].
    * Default: 0.0001.
    */
  val tol : DoubleParam = new DoubleParam(this, "local_lloyd_tol",
    "Tolerance for change in ssd for early stopping in local kmeans. Must be in range [0, 1].",
    ParamValidators.inRange(0.0, 1.0))
  def getTol: Double = $(tol)

  /**
    * The number of trials of the local kmeans algorithm. The output with best loss will be chosen.
    * Must be > 0 or "auto".
    * Default: "auto".
    */
  val trialNum : Param[String] = new Param(this, "local_lloyd_num_trials",
    "The number of trials of the local kmeans algorithm. Must be > 0 or 'auto'.",
    autoOrAboveParamValidator(0, false))
  def getTrialNum: String = $(trialNum)

  /**
    * The local initialization algorithm to choose centroids. Must be "random" or "kmeans++".
    * Default: "kmeans++".
    */
  val localInitMethod : Param[String] = new Param(this, "local_lloyd_init_method",
    "The local initialization algorithm to choose centroids. Supported options: 'random' and " +
      "'kmeans++'", ParamValidators.inArray(Array("random", "kmeans++")))
  def getLocalInitMethod: String = $(localInitMethod)

  /**
    * The weight decaying rate of each point. 0 means no decay at all. Must be >= 0.
    * Default: 0.
    */
  val halflifeTime : IntParam = new IntParam(this, "half_life_time_size",
    "The weight decaying rate of each point. Must be >= 0.",
    ParamValidators.gtEq(0))
  def getHalflifeTime: Int = $(halflifeTime)

  /**
    * The number of passes done over the training data. Must be > 0.
    * Default: 1.
    */
  val epochs : IntParam = new IntParam(this, "epochs",
    "The number of passes done over the training data. Must be > 0.",
    ParamValidators.gtEq(1))
  def getEpochs: Int = $(epochs)

  /**
    * The factor of extra centroids to create. The number of initial centroids equals
    * centerFactor * k. Must be > 0 or "auto".
    * Default: "auto".
    */
  val centerFactor : Param[String] = new Param(this, "extra_center_factor",
    "The factor of extra centroids to create. Must be > 0 or 'auto'",
    autoOrAboveParamValidator(0, false))
  def getCenterFactor: String = $(centerFactor)

  /** Metric to be used for scoring the model. String of comma separated metrics.
    * Support metrics are "msd" and "ssd".
    * "msd" Means Square Error, "ssd": Sum of square distance
    * Default = "msd"
    */
  val evalMetrics: Param[String] = new Param(this, "eval_metrics",
    "Metric to be used for scoring the model. String of comma separated metrics. " +
      "Support metrics are 'msd' and 'ssd'." +
      "'msd' Means Square Error, 'ssd': Sum of square distance" +
      "is usually set automatically, depending on some other parameters", evalMetricsValidator)
  def getEvalMetrics: String = $(evalMetrics).stripPrefix("[").stripSuffix("]")

  private def evalMetricsValidator: String => Boolean = {
    (value: String) => value.stripPrefix("[").stripSuffix("]").split(",")
      .map(metric => evalMetricsValues.contains(metric.trim)).reduce(_ && _)
  }

  private val evalMetricsValues = Array("msd", "ssd")
}

object KMeansSageMakerEstimator {
  val algorithmName = "kmeans"
  val algorithmTag = "1"
  val regionAccountMap = SagerMakerRegionAccountMaps.AlgorithmsAccountMap
}

/**
  * A [[SageMakerEstimator]] that runs a K-Means Clustering training job on Amazon SageMaker upon a
  * call to fit() on a DataFrame and returns a [[SageMakerModel]]
  * that can be used to transform a DataFrame using the hosted K-Means model. K-Means Clustering
  * is useful for grouping similar examples in your dataset.
  *
  * Amazon SageMaker K-Means clustering trains on RecordIO-encoded Amazon Record protobuf data.
  * SageMaker Spark writes a DataFrame to S3 by selecting a column of Vectors named "features"
  * and, if present, a column of Doubles named "label". These names are configurable by passing
  * a map with entries in trainingSparkDataFormatOptions with key "labelColumnName" or
  * "featuresColumnName", with values corresponding to the desired label and features columns.
  *
  * For inference, the SageMakerModel returned by fit() by the KMeansSageMakerEstimator
  * uses [[ProtobufRequestRowSerializer]] to serialize Rows into
  * RecordIO-encoded Amazon Record protobuf messages for inference, by default selecting
  * the column named "features" expected to contain a Vector of Doubles.
  *
  * Inferences made against an Endpoint hosting a K-Means model contain
  * a "closest_cluster" field and a "distance_to_cluster" field, both
  * appended to the input DataFrame as columns of Double.
  *
  * @param sagemakerRole The SageMaker TrainingJob and Hosting IAM Role. Used by a SageMaker to
  *                      access S3 and ECR resources. SageMaker hosted Endpoints instances
  *                      launched by this Estimator run with this role.
  * @param trainingInstanceType The SageMaker TrainingJob Instance Type to use
  * @param trainingInstanceCount The number of instances of instanceType to run an
  *                              SageMaker Training Job with
  * @param endpointInstanceType The SageMaker Endpoint Confing instance type
  * @param endpointInitialInstanceCount The SageMaker Endpoint Config minimum number of instances
  *                                     that can be used to host modelImage
  * @param requestRowSerializer Serializes Spark DataFrame [[Row]]s for transformation by Models
  *                             built from this Estimator.
  * @param responseRowDeserializer Deserializes an Endpoint response into a series of [[Row]]s.
  * @param trainingInputS3DataPath An S3 location to upload SageMaker Training Job input data to.
  * @param trainingOutputS3DataPath An S3 location for SageMaker to store Training Job output
  *                                 data to.
  * @param trainingInstanceVolumeSizeInGB The EBS volume size in gigabytes of each instance.
  * @param trainingProjectedColumns The columns to project from the Dataset being fit before
  *                                 training. If an Optional.empty is passed then no specific
  *                                 projection will occur and all columns will be serialized.
  * @param trainingChannelName The SageMaker Channel name to input serialized Dataset fit input to
  * @param trainingContentType The MIME type of the training data.
  * @param trainingS3DataDistribution The SageMaker Training Job S3 data distribution scheme.
  * @param trainingSparkDataFormat The Spark Data Format name used to serialize the Dataset being
  *                                fit for input to SageMaker.
  * @param trainingSparkDataFormatOptions The Spark Data Format Options used during serialization of
  *                                       the Dataset being fit.
  * @param trainingInputMode The SageMaker Training Job Channel input mode.
  * @param trainingCompressionCodec The type of compression to use when serializing the Dataset
  *                                 being fit for input to SageMaker.
  * @param trainingMaxRuntimeInSeconds A SageMaker Training Job Termination Condition
  *                                    MaxRuntimeInHours.
  * @param trainingKmsKeyId A KMS key ID for the Output Data Source
  * @param modelEnvironmentVariables The environment variables that SageMaker will set on the model
  *                                  container during execution.
  * @param endpointCreationPolicy Defines how a SageMaker Endpoint referenced by a
  *                               SageMakerModel is created.
  * @param sagemakerClient Amazon SageMaker client. Used to send CreateTrainingJob, CreateModel,
  *                        and CreateEndpoint requests.
  * @param region The region in which to run the algorithm. If not specified, gets the region from
  *               the DefaultAwsRegionProviderChain.
  * @param s3Client AmazonS3. Used to create a bucket for staging SageMaker Training Job input
  *                 and/or output if either are set to S3AutoCreatePath.
  * @param stsClient AmazonSTS. Used to resolve the account number when creating staging
  *                  input / output buckets.
  * @param modelPrependInputRowsToTransformationRows Whether the transformation result on Models
  *        built by this Estimator should also include the input Rows. If true, each output Row
  *        is formed by a concatenation of the input Row with the corresponding Row produced by
  *        SageMaker Endpoint invocation, produced by responseRowDeserializer.
  *        If false, each output Row is just taken from responseRowDeserializer.
  * @param deleteStagingDataAfterTraining Whether to remove the training data on s3 after training
  *                                       is complete or failed.
  * @param namePolicyFactory The [[NamePolicyFactory]] to use when naming SageMaker entities
  *        created during fit
  * @param uid The unique identifier of this Estimator. Used to represent this stage in Spark
  *            ML pipelines.
  */
class KMeansSageMakerEstimator(
      override val sagemakerRole : IAMRoleResource = IAMRoleFromConfig(),
      override val trainingInstanceType : String,
      override val trainingInstanceCount : Int,
      override val endpointInstanceType : String,
      override val endpointInitialInstanceCount : Int,
      override val requestRowSerializer : RequestRowSerializer =
        new ProtobufRequestRowSerializer(),
      override val responseRowDeserializer : ResponseRowDeserializer =
        new KMeansProtobufResponseRowDeserializer(),
      override val trainingInputS3DataPath : S3Resource = S3AutoCreatePath(),
      override val trainingOutputS3DataPath : S3Resource = S3AutoCreatePath(),
      override val trainingInstanceVolumeSizeInGB : Int = 1024,
      override val trainingProjectedColumns : Option[List[String]] = None,
      override val trainingChannelName : String = "train",
      override val trainingContentType: Option[String] = None,
      override val trainingS3DataDistribution : String = S3DataDistribution.ShardedByS3Key.toString,
      override val trainingSparkDataFormat : String = "sagemaker",
      override val trainingSparkDataFormatOptions : Map[String, String] = Map(),
      override val trainingInputMode : String = TrainingInputMode.File.toString,
      override val trainingCompressionCodec : Option[String] = None,
      override val trainingMaxRuntimeInSeconds : Int = 24 * 60 * 60,
      override val trainingKmsKeyId : Option[String] = None,
      override val modelEnvironmentVariables : Map[String, String] = Map(),
      override val endpointCreationPolicy : EndpointCreationPolicy =
        EndpointCreationPolicy.CREATE_ON_CONSTRUCT,
      override val sagemakerClient : AmazonSageMaker
        = AmazonSageMakerClientBuilder.defaultClient,
      val region : Option[String] = None,
      override val s3Client : AmazonS3 = AmazonS3ClientBuilder.defaultClient(),
      override val stsClient : AWSSecurityTokenService =
        AWSSecurityTokenServiceClientBuilder.defaultClient(),
      override val modelPrependInputRowsToTransformationRows : Boolean = true,
      override val deleteStagingDataAfterTraining : Boolean = true,
      override val namePolicyFactory : NamePolicyFactory = new RandomNamePolicyFactory(),
      override val uid : String = Identifiable.randomUID("sagemaker"))
  extends SageMakerEstimator(
    trainingImage = SageMakerImageURIProvider.getImage(
      region.getOrElse(new DefaultAwsRegionProviderChain().getRegion),
      KMeansSageMakerEstimator.regionAccountMap,
      KMeansSageMakerEstimator.algorithmName, KMeansSageMakerEstimator.algorithmTag),
    modelImage = SageMakerImageURIProvider.getImage(
      region.getOrElse(new DefaultAwsRegionProviderChain().getRegion),
      KMeansSageMakerEstimator.regionAccountMap,
      KMeansSageMakerEstimator.algorithmName, KMeansSageMakerEstimator.algorithmTag),
    sagemakerRole,
    trainingInstanceType,
    trainingInstanceCount,
    endpointInstanceType,
    endpointInitialInstanceCount,
    requestRowSerializer,
    responseRowDeserializer,
    trainingInputS3DataPath,
    trainingOutputS3DataPath,
    trainingInstanceVolumeSizeInGB,
    trainingProjectedColumns,
    trainingChannelName,
    trainingContentType,
    trainingS3DataDistribution,
    trainingSparkDataFormat,
    trainingSparkDataFormatOptions,
    trainingInputMode,
    trainingCompressionCodec,
    trainingMaxRuntimeInSeconds,
    trainingKmsKeyId,
    modelEnvironmentVariables,
    endpointCreationPolicy,
    sagemakerClient,
    s3Client,
    stsClient,
    modelPrependInputRowsToTransformationRows,
    deleteStagingDataAfterTraining,
    namePolicyFactory,
    uid) with KMeansParams {

  setDefault(miniBatchSize -> 5000)

  def setK(value : Int) : this.type = set(k, value)

  def setFeatureDim(value : Int) : this.type = set(featureDim, value)

  def setMiniBatchSize(value : Int) : this.type = set(miniBatchSize, value)

  def setMaxIter(value : Int) : this.type = set(maxIter, value)

  def setTol(value : Double) : this.type = set(tol, value)

  def setLocalInitMethod(value : String) : this.type = set(localInitMethod, value)

  def setHalflifeTime(value : Int) : this.type = set(halflifeTime, value)

  def setEpochs(value : Int) : this.type = set(epochs, value)

  def setInitMethod(value : String) : this.type = set(initMethod, value)

  def setCenterFactor(value : String) : this.type = set(centerFactor, value)

  def setCenterFactor(value : Int) : this.type = set(centerFactor, value.toString)

  def setTrialNum(value : String) : this.type = set(trialNum, value)

  def setTrialNum(value : Int) : this.type = set(trialNum, value.toString)

  def setEvalMetrics(value : String) : this.type = set(evalMetrics, "[" + value + "]")

  // Check whether required hyper-parameters are set
  override def transformSchema(schema: StructType): StructType = {
    $(featureDim)
    $(k)
    $(miniBatchSize)
    super.transformSchema(schema)
  }

}
