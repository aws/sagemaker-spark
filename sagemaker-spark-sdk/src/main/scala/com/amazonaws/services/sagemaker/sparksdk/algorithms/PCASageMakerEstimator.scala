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
import com.amazonaws.services.sagemaker.sparksdk.transformation.deserializers.PCAProtobufResponseRowDeserializer
import com.amazonaws.services.sagemaker.sparksdk.transformation.serializers.ProtobufRequestRowSerializer


/**
  * Common params for [[PCASageMakerEstimator]] with accessors
  */
private[algorithms] trait PCAParams extends SageMakerAlgorithmParams {

  /**
    * Number of principal components. Required.
    */
  val numComponents : IntParam = new IntParam(this, "num_components",
    "Number of principal components. " +
    "Must be > 0",
    ParamValidators.gt(0))
  def getNumComponents: Int = $(numComponents)

  /**
    * PCA algorithm. Supported options: "regular", "stable", and "randomized".
    * Default: "regular".
    */
  val algorithmMode : Param[String] = new Param(this, "algorithm_mode",
    "PCA algorithm. " +
    "Supported options: 'regular', 'stable', and 'randomized'.",
    ParamValidators.inArray(Array("regular", "stable", "randomized")))
  def getAlgorithmMode: String = $(algorithmMode)

  /**
    * Whether to subtract the mean during training and inference
    * Default: True
    */
  val subtractMean : Param[String] = new Param(this, "subtract_mean",
    "If true, the data will be unbiased both during train and at inference",
    ParamValidators.inArray(Array("True", "False")))
  def getSubtractMean: Boolean = parseTrueAndFalse(subtractMean)

  /**
    * Number of extra components to compute. Must be -1 or > 0.
    * Valid for "randomized" mode. Ignored by other modes.
    * Initializes a random matrix for covariance computation independent from the desired
    * num_components. As it grows larger, the solution is more accurate but the runtime and memory
    * consumption increase linearly.
    * Default: -1
    */
  val extraComponents: IntParam = new IntParam(this, "extra_components",
    "Number of extra components to compute",
    (extraComponents: Int) => extraComponents == -1 || extraComponents > 0)
  def getExtraComponents: Integer = $(extraComponents)
}

object PCASageMakerEstimator {
  val algorithmName = "pca"
  val algorithmTag = "1"
  val regionAccountMap = SagerMakerRegionAccountMaps.AlgorithmsAccountMap
}

/**
  * A [[SageMakerEstimator]] that runs a PCA training job in SageMaker and returns
  * a [[SageMakerModel]] that can be used to transform a DataFrame using
  * the hosted PCA model. PCA, or Principal Component Analysis, is useful
  * for reducing the dimensionality of data before training with another algorithm.
  *
  * Amazon SageMaker PCA trains on RecordIO-encoded Amazon Record protobuf data.
  * SageMaker Spark writes a DataFrame to S3 by selecting a column of Vectors named "features"
  * and, if present, a column of Doubles named "label". These names are configurable by passing
  * a map with entries in trainingSparkDataFormatOptions with key "labelColumnName" or
  * "featuresColumnName", with values corresponding to the desired label and features columns.
  *
  * PCASageMakerEstimator uses [[ProtobufRequestRowSerializer]] to serialize Rows into
  * RecordIO-encoded Amazon Record protobuf messages for inference, by default selecting
  * the column named "features" expected to contain a Vector of Doubles.
  *
  * Inferences made against an Endpoint hosting a PCA model contain
  * a "projection" field appended to the input DataFrame as a Dense Vector of Doubles.
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
class PCASageMakerEstimator(
                       override val sagemakerRole : IAMRoleResource = IAMRoleFromConfig(),
                       override val trainingInstanceType : String,
                       override val trainingInstanceCount : Int,
                       override val endpointInstanceType : String,
                       override val endpointInitialInstanceCount : Int,
                       override val requestRowSerializer : RequestRowSerializer =
                       new ProtobufRequestRowSerializer(),
                       override val responseRowDeserializer : ResponseRowDeserializer =
                       new PCAProtobufResponseRowDeserializer(),
                       override val trainingInputS3DataPath : S3Resource = S3AutoCreatePath(),
                       override val trainingOutputS3DataPath : S3Resource = S3AutoCreatePath(),
                       override val trainingInstanceVolumeSizeInGB : Int = 1024,
                       override val trainingProjectedColumns : Option[List[String]] = None,
                       override val trainingChannelName : String = "train",
                       override val trainingContentType: Option[String] = None,
                       override val trainingS3DataDistribution : String
                         = S3DataDistribution.ShardedByS3Key.toString,
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
                       override val namePolicyFactory : NamePolicyFactory
                         = new RandomNamePolicyFactory(),
                       override val uid : String = Identifiable.randomUID("sagemaker"))
  extends SageMakerEstimator(
    trainingImage = SageMakerImageURIProvider.getImage(
      region.getOrElse(new DefaultAwsRegionProviderChain().getRegion),
      PCASageMakerEstimator.regionAccountMap,
      PCASageMakerEstimator.algorithmName, PCASageMakerEstimator.algorithmTag),
    modelImage = SageMakerImageURIProvider.getImage(
      region.getOrElse(new DefaultAwsRegionProviderChain().getRegion),
      PCASageMakerEstimator.regionAccountMap,
      PCASageMakerEstimator.algorithmName, PCASageMakerEstimator.algorithmTag),
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
    uid) with PCAParams {

  setDefault(miniBatchSize -> 1000)

  def setNumComponents(value : Int) : this.type = set(numComponents, value)

  def setFeatureDim(value : Int) : this.type = set(featureDim, value)

  def setMiniBatchSize(value : Int) : this.type = set(miniBatchSize, value)

  def setAlgorithmMode(value: String) : this.type = set(algorithmMode, value)

  def setSubtractMean(value: Boolean) : this.type = set(subtractMean, value match {
    case true => "True"
    case false => "False"
  })

  def setExtraComponents(value: Int) : this.type = set(extraComponents, value)

  // Check whether required hyper-parameters are set
  override def transformSchema(schema: StructType): StructType = {
    $(featureDim)
    $(numComponents)
    $(miniBatchSize)
    super.transformSchema(schema)
  }

}
