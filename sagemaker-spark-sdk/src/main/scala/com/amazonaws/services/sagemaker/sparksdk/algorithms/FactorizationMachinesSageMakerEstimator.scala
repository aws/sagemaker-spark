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
import com.amazonaws.services.sagemaker.sparksdk.transformation.deserializers.{FactorizationMachinesBinaryClassifierProtobufResponseRowDeserializer, FactorizationMachinesRegressorProtobufResponseRowDeserializer}
import com.amazonaws.services.sagemaker.sparksdk.transformation.serializers.ProtobufRequestRowSerializer

/**
  * Common params for [[LinearLearnerSageMakerEstimator]] with accessors
  */
private[algorithms] trait FactorizationMachinesParams extends SageMakerAlgorithmParams {

  /**
    * Dimensionality of factorization. Must be > 0.
    * Required
    */
  val numFactors : IntParam = new IntParam(this, "num_factors",
    "Dimensionality of factorization. Must be > 0.",
    ParamValidators.gt(0))
  def getNumFactors: Int = $(numFactors)

  /**
    * Whether training is for binary classification or regression.
    * Supported options: "binary_classifier", and "regressor".
    * Required
    */
  private[algorithms] val predictorType : Param[String] = new Param(this, "predictor_type",
    "Whether training is for binary classification or regression. " +
      "Supported options: 'binary_classifier', and 'regressor'.",
    ParamValidators.inArray(Array("binary_classifier", "regressor")))

  /**
    * Number of training epochs to run. Must be > 0.
    * Default: 1
    */
  val epochs : IntParam = new IntParam(this, "epochs",
    "Number of training epochs to run. Must be > 0.",
    ParamValidators.gt(0))
  def getEpochs: Int = $(epochs)

  /**
    * Clip the gradient by projecting onto the box [-clip_gradient, +clip_gradient].
    */
  val clipGradient : DoubleParam = new DoubleParam(this, "clip_gradient",
    "Clip the gradient by projecting onto the box [-clip_gradient, +clip_gradient].")
  def getClipGradient: Double = $(clipGradient)

  /**
    * Small value to avoid division by 0.
    */
  val eps : DoubleParam = new DoubleParam(this, "eps",
    "Small value to avoid division by 0.")
  def getEps: Double = $(eps)

  /**
    * If set, multiplies the gradient with rescale_grad before updating.
    * Often choose to be 1.0/batch_size.
    */
  val rescaleGrad : DoubleParam = new DoubleParam(this, "rescale_grad",
    "Multiplies the gradient with this value before updating")
  def getRescaleGrad: Double = $(rescaleGrad)

  /**
    * Non-negative learning rate for the bias term.
    * Suggested value range: [1e-8, 512].
    * Default: 0.1
    */
  val biasLr : DoubleParam = new DoubleParam(this, "bias_lr",
    "Multiplies the gradient with this value before updating. Must be > 0",
    ParamValidators.gt(0))
  def getBiasLr: Double = $(biasLr)

  /**
    * Non-negative learning rate for linear terms.
    * Suggested value range: [1e-8, 512].
    * Default: 0.001
    */
  val linearLr : DoubleParam = new DoubleParam(this, "linear_lr",
    "Learning rate for linear terms. Must be > 0",
    ParamValidators.gt(0))
  def getLinearLr: Double = $(linearLr)

  /**
    * Non-negative learning rate for factorization terms.
    * Suggested value range: [1e-8, 512].
    * Default: 0.0001
    */
  val factorsLr : DoubleParam = new DoubleParam(this, "factors_lr",
    "Learning rate for factorization terms. Must be > 0",
    ParamValidators.gt(0))
  def getFactorsLr: Double = $(factorsLr)

  /**
    * Non-negative weight decay for the bias term.
    * Suggested value range: [1e-8, 512].
    * Default: 0.01
    */
  val biasWd : DoubleParam = new DoubleParam(this, "bias_wd",
    "Weight decay for the bias term. Must be > 0",
    ParamValidators.gt(0))
  def getBiasWd: Double = $(biasWd)

  /**
    * Non-negative weight decay for linear terms.
    * Suggested value range: [1e-8, 512].
    * Default: 0.001
    */
  val linearWd : DoubleParam = new DoubleParam(this, "linear_wd",
    "Weight decay for linear terms. Must be > 0",
    ParamValidators.gt(0))
  def getLinearWd: Double = $(linearWd)

  /**
    * Non-negative weight decay for factorization terms.
    * Suggested value range: [1e-8, 512].
    * Default: 0.00001
    */
  val factorsWd : DoubleParam = new DoubleParam(this, "factors_wd",
    "Weight decay for factorization terms. Must be > 0",
    ParamValidators.gt(0))
  def getFactorsWd: Double = $(factorsWd)

  /**
    * Initialization method for the bias term.
    * Supported options: "normal", "uniform" or "constant".
    * uniform: random weights sampled uniformly from range [-bias_init_scale, +bias_init_scale]
    * normal: random weights from normal distribution with mean 0 and std dev "bias_init_sigma"
    * constant: weights initialized to "bias_init_value"
    * Default: "normal".
    */
  val biasInitMethod : Param[String] = new Param(this, "bias_init_method",
    "Initialization method for the bias supports 'normal', 'uniform' and 'constant'.",
    ParamValidators.inArray(Array("normal", "uniform", "constant")))
  def getBiasInitMethod: String = $(biasInitMethod)

  /**
    * Non-negative range for initialization of the bias term when "bias_init_method" is "uniform".
    * Suggested value range: [1e-8, 512].
    */
  val biasInitScale : DoubleParam = new DoubleParam(this, "bias_init_scale",
    "Range for bias term uniform initialization. Must be > 0.",
    ParamValidators.gt(0))
  def getBiasInitScale: Double = $(biasInitScale)

  /**
    * Non-negative standard deviation of the bias term when "bias_init_method" is "normal".
    * Suggested value range: [1e-8, 512].
    * Default: 0.01.
    */
  val biasInitSigma : DoubleParam = new DoubleParam(this, "bias_init_sigma",
    "Standard deviation for initialization of the bias terms. Must be > 0.",
    ParamValidators.gt(0))
  def getBiasInitSigma: Double = $(biasInitSigma)

  /**
    * Initial value of the bias term when "bias_init_method" is "constant".
    * Suggested value range: [1e-8, 512]
    */
  val biasInitValue : DoubleParam = new DoubleParam(this, "bias_init_value",
    "Initial value for the bias term.")
  def getBiasInitValue: Double = $(biasInitValue)

  /**
    * Initialization method for linear term.
    * Supported options: "normal", "uniform" or "constant".
    * uniform: random weights sampled uniformly from range [-linear_init_scale, +linear_init_scale]
    * normal: random weights from normal distribution with mean 0 and std dev "linear_init_sigma"
    * constant: weights initialized to "linear_init_value"
    * Default: "normal".
    */
  val linearInitMethod : Param[String] = new Param(this, "linear_init_method",
    "Initialization method for linear term. Supported options: 'normal', 'uniform' and 'constant'.",
    ParamValidators.inArray(Array("normal", "uniform", "constant")))
  def getLinearInitMethod: String = $(linearInitMethod)

  /**
    * Non-negative range for initialization of linear term when "linear_init_method" is "uniform".
    * Suggested value range: [1e-8, 512].
    */
  val linearInitScale : DoubleParam = new DoubleParam(this, "linear_init_scale",
    "Range for linear term uniform initialization. Must be > 0.",
    ParamValidators.gt(0))
  def getLinearInitScale: Double = $(linearInitScale)

  /**
    * Non-negative standard deviation of linear term when "linear_init_method" is "normal".
    * Suggested value range: [1e-8, 512].
    * Default: 0.01.
    */
  val linearInitSigma : DoubleParam = new DoubleParam(this, "linear_init_sigma",
    "Standard deviation for initialization of linear terms. Must be > 0.",
    ParamValidators.gt(0))
  def getLinearInitSigma: Double = $(linearInitSigma)

  /**
    * Initial value of linear term when "linear_init_method" is "constant".
    * Suggested value range: [1e-8, 512]
    */
  val linearInitValue : DoubleParam = new DoubleParam(this, "linear_init_value",
    "Initial value for linear term.")
  def getLinearInitValue: Double = $(linearInitValue)


  /**
    * Initialization method for factorization terms.
    * Supported options: "normal", "uniform" or "constant".
    * uniform: random weights sampled uniformly from [-factors_init_scale, +factors_init_scale]
    * normal: random weights from normal distribution with mean 0 and std dev "factors_init_sigma"
    * constant: weights initialized to "factors_init_value"
    * Default: "normal".
    */
  val factorsInitMethod : Param[String] = new Param(this, "factors_init_method",
    "Initialization method for factorization terms supports 'normal', 'uniform' and 'constant'.",
    ParamValidators.inArray(Array("normal", "uniform", "constant")))
  def getFactorsInitMethod: String = $(factorsInitMethod)

  /**
    * Non-negative range for factorization terms initialization in "uniform" (factors_init_method).
    * Suggested value range: [1e-8, 512].
    */
  val factorsInitScale : DoubleParam = new DoubleParam(this, "factors_init_scale",
    "Range for factorization terms uniform initialization. Must be > 0.",
    ParamValidators.gt(0))
  def getFactorsInitScale: Double = $(factorsInitScale)

  /**
    * Non-negative standard deviation of factorization terms when "factors_init_method" is "normal".
    * Suggested value range: [1e-8, 512].
    * Default: 0.001.
    */
  val factorsInitSigma : DoubleParam = new DoubleParam(this, "factors_init_sigma",
    "Standard deviation for initialization of factorization terms. Must be > 0.",
    ParamValidators.gt(0))
  def getFactorsInitSigma: Double = $(factorsInitSigma)

  /**
    * Initial value of factorization terms when "factors_init_method" is "constant".
    * Suggested value range: [1e-8, 512]
    */
  val factorsInitValue : DoubleParam = new DoubleParam(this, "factors_init_value",
    "Initial value for factorization terms.")
  def getFactorsInitValue: Double = $(factorsInitValue)
}

object FactorizationMachinesSageMakerEstimator {
  val algorithmName = "factorization-machines"
  val algorithmTag = "1"
  val regionAccountMap = SagerMakerRegionAccountMaps.AlgorithmsAccountMap
}

/**
  * A [[SageMakerEstimator]] that runs a Factorization Machines training job in "binary classifier"
  * mode in SageMaker and returns a [[SageMakerModel]] that can be used to transform a DataFrame
  * using the hosted Factorization Machines model. The Linear Learner Binary Classifier is
  * useful for classifying examples into one of two classes.
  *
  * Amazon SageMaker Factorization Machines trains on RecordIO-encoded Amazon Record protobuf data.
  * SageMaker Spark writes a DataFrame to S3 by selecting a column of Vectors named "features"
  * and, if present, a column of Doubles named "label". These names are configurable by passing
  * a map with entries in trainingSparkDataFormatOptions with key "labelColumnName" or
  * "featuresColumnName", with values corresponding to the desired label and features columns.
  *
  * Inferences made against an Endpoint hosting a Factorization Machines Binary classifier
  * model contain a "score" field and a "predicted_label" field, both appended to the input
  * DataFrame as Doubles.
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
class FactorizationMachinesBinaryClassifier(
           override val sagemakerRole : IAMRoleResource = IAMRoleFromConfig(),
           override val trainingInstanceType : String,
           override val trainingInstanceCount : Int,
           override val endpointInstanceType : String,
           override val endpointInitialInstanceCount : Int,
           override val requestRowSerializer : RequestRowSerializer =
           new ProtobufRequestRowSerializer(),
           override val responseRowDeserializer : ResponseRowDeserializer =
           new FactorizationMachinesBinaryClassifierProtobufResponseRowDeserializer(),
           override val trainingInputS3DataPath : S3Resource = S3AutoCreatePath(),
           override val trainingOutputS3DataPath : S3Resource = S3AutoCreatePath(),
           override val trainingInstanceVolumeSizeInGB : Int = 1024,
           override val trainingProjectedColumns : Option[List[String]] = None,
           override val trainingChannelName : String = "train",
           override val trainingContentType: Option[String] = None,
           override val trainingS3DataDistribution : String =
           S3DataDistribution.ShardedByS3Key.toString,
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
           override val region : Option[String] = None,
           override val s3Client : AmazonS3 = AmazonS3ClientBuilder.defaultClient(),
           override val stsClient : AWSSecurityTokenService =
           AWSSecurityTokenServiceClientBuilder.defaultClient(),
           override val modelPrependInputRowsToTransformationRows : Boolean = true,
           override val deleteStagingDataAfterTraining : Boolean = true,
           override val namePolicyFactory : NamePolicyFactory = new RandomNamePolicyFactory(),
           override val uid : String = Identifiable.randomUID("sagemaker"))
  extends FactorizationMachinesSageMakerEstimator(
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
    region,
    s3Client,
    stsClient,
    modelPrependInputRowsToTransformationRows,
    deleteStagingDataAfterTraining,
    namePolicyFactory,
    uid) with FactorizationMachinesParams {

  setDefault(predictorType -> "binary_classifier")
}


/**
  * A [[SageMakerEstimator]] that runs a Factorization Machines training job in "regressor" mode in
  * SageMaker and returns a [[SageMakerModel]] that can be used to transform a DataFrame using
  * the hosted Factorization Machines model. The Factorization Machines Regressor is useful for
  * predicting a real-valued label from training examples.
  *
  * Amazon SageMaker Factorization Machines trains on RecordIO-encoded Amazon Record protobuf data.
  * SageMaker Spark writes a DataFrame to S3 by selecting a column of Vectors named "features"
  * and, if present, a column of Doubles named "label". These names are configurable by passing
  * a map with entries in trainingSparkDataFormatOptions with key "labelColumnName" or
  * "featuresColumnName", with values corresponding to the desired label and features columns.
  *
  * For inference against a hosted Endpoint, the SageMakerModel returned by fit() by
  * Linear Learner uses [[ProtobufRequestRowSerializer]] to serialize Rows into
  * RecordIO-encoded Amazon Record protobuf messages, by default selecting
  * the column named "features" expected to contain a Vector of Doubles.
  *
  * Inferences made against an Endpoint hosting a Factorization Machines Regressor model contain
  * a "score" field appended to the input DataFrame as a Double
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
class FactorizationMachinesRegressor(
            override val sagemakerRole : IAMRoleResource = IAMRoleFromConfig(),
            override val trainingInstanceType : String,
            override val trainingInstanceCount : Int,
            override val endpointInstanceType : String,
            override val endpointInitialInstanceCount : Int,
            override val requestRowSerializer : RequestRowSerializer =
            new ProtobufRequestRowSerializer(),
            override val responseRowDeserializer : ResponseRowDeserializer =
            new FactorizationMachinesRegressorProtobufResponseRowDeserializer(),
            override val trainingInputS3DataPath : S3Resource = S3AutoCreatePath(),
            override val trainingOutputS3DataPath : S3Resource = S3AutoCreatePath(),
            override val trainingInstanceVolumeSizeInGB : Int = 1024,
            override val trainingProjectedColumns : Option[List[String]] = None,
            override val trainingChannelName : String = "train",
            override val trainingContentType: Option[String] = None,
            override val trainingS3DataDistribution : String =
            S3DataDistribution.ShardedByS3Key.toString,
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
            override val region : Option[String] = None,
            override val s3Client : AmazonS3 = AmazonS3ClientBuilder.defaultClient(),
            override val stsClient : AWSSecurityTokenService =
            AWSSecurityTokenServiceClientBuilder.defaultClient(),
            override val modelPrependInputRowsToTransformationRows : Boolean = true,
            override val deleteStagingDataAfterTraining : Boolean = true,
            override val namePolicyFactory : NamePolicyFactory
            = new RandomNamePolicyFactory(),
            override val uid : String = Identifiable.randomUID("sagemaker"))
  extends FactorizationMachinesSageMakerEstimator(
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
    region,
    s3Client,
    stsClient,
    modelPrependInputRowsToTransformationRows,
    deleteStagingDataAfterTraining,
    namePolicyFactory,
    uid) with FactorizationMachinesParams {
  setDefault(predictorType -> "regressor")
}

/**
  * A [[SageMakerEstimator]] that runs a Factorization Machines training job in SageMaker and
  * returns a [[SageMakerModel]] that can be used to transform a DataFrame using
  * the hosted Factorization Machines model. This algorithm can be run in either regression
  * (see [[FactorizationMachinesRegressor]]) or binary classification
  * (see [[FactorizationMachinesBinaryClassifier]]) modes.
  *
  * Amazon SageMaker Factorization Machines trains on RecordIO-encoded Amazon Record protobuf data.
  * SageMaker Spark writes a DataFrame to S3 by selecting a column of Vectors named "features"
  * and, if present, a column of Doubles named "label". These names are configurable by passing
  * a map with entries in trainingSparkDataFormatOptions with key "labelColumnName" or
  * "featuresColumnName", with values corresponding to the desired label and features columns.
  *
  * For inference against a hosted Endpoint, the SageMakerModel returned by fit() by
  * Factorization MAchines uses [[ProtobufRequestRowSerializer]] to serialize Rows into
  * RecordIO-encoded Amazon Record protobuf messages, by default selecting
  * the column named "features" expected to contain a Vector of Doubles.
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
private[algorithms] class FactorizationMachinesSageMakerEstimator(
            override val sagemakerRole : IAMRoleResource = IAMRoleFromConfig(),
            override val trainingInstanceType : String,
            override val trainingInstanceCount : Int,
            override val endpointInstanceType : String,
            override val endpointInitialInstanceCount : Int,
            override val requestRowSerializer : RequestRowSerializer,
            override val responseRowDeserializer : ResponseRowDeserializer,
            override val trainingInputS3DataPath : S3Resource = S3AutoCreatePath(),
            override val trainingOutputS3DataPath : S3Resource = S3AutoCreatePath(),
            override val trainingInstanceVolumeSizeInGB : Int = 1024,
            override val trainingProjectedColumns : Option[List[String]] = None,
            override val trainingChannelName : String = "train",
            override val trainingContentType: Option[String] = None,
            override val trainingS3DataDistribution : String =
            S3DataDistribution.ShardedByS3Key.toString,
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
      FactorizationMachinesSageMakerEstimator.regionAccountMap,
      FactorizationMachinesSageMakerEstimator.algorithmName,
      FactorizationMachinesSageMakerEstimator.algorithmTag),
    modelImage = SageMakerImageURIProvider.getImage(
      region.getOrElse(new DefaultAwsRegionProviderChain().getRegion),
      FactorizationMachinesSageMakerEstimator.regionAccountMap,
      FactorizationMachinesSageMakerEstimator.algorithmName,
      FactorizationMachinesSageMakerEstimator.algorithmTag),
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
    uid) with FactorizationMachinesParams {

  def setFeatureDim(value: Int): this.type = set(featureDim, value)

  def setMiniBatchSize(value: Int): this.type = set(miniBatchSize, value)

  def setNumFactors(value: Int): this.type = set(numFactors, value)

  def setEpochs(value: Int): this.type = set(epochs, value)

  def setClipGradient(value: Double): this.type = set(clipGradient, value)

  def setEps(value: Double): this.type = set(eps, value)

  def setRescaleGrad(value: Double): this.type = set(rescaleGrad, value)

  def setBiasLr(value: Double): this.type = set(biasLr, value)

  def setLinearLr(value: Double): this.type = set(linearLr, value)

  def setFactorsLr(value: Double): this.type = set(factorsLr, value)

  def setBiasWd(value: Double): this.type = set(biasWd, value)

  def setLinearWd(value: Double): this.type = set(linearWd, value)

  def setFactorsWd(value: Double): this.type = set(factorsWd, value)

  def setBiasInitMethod(value: String): this.type = set(biasInitMethod, value)

  def setBiasInitScale(value: Double): this.type = set(biasInitScale, value)

  def setBiasInitSigma(value: Double): this.type = set(biasInitSigma, value)

  def setBiasInitValue(value: Double): this.type = set(biasInitValue, value)

  def setLinearInitMethod(value: String): this.type = set(linearInitMethod, value)

  def setLinearInitScale(value: Double): this.type = set(linearInitScale, value)

  def setLinearInitSigma(value: Double): this.type = set(linearInitSigma, value)

  def setLinearInitValue(value: Double): this.type = set(linearInitValue, value)

  def setFactorsInitMethod(value: String): this.type = set(factorsInitMethod, value)

  def setFactorsInitScale(value: Double): this.type = set(factorsInitScale, value)

  def setFactorsInitSigma(value: Double): this.type = set(factorsInitSigma, value)

  def setFactorsInitValue(value: Double): this.type = set(factorsInitValue, value)

  // Check whether required hyper-parameters are set
  override def transformSchema(schema: StructType): StructType = {
    $(featureDim)
    $(predictorType)
    $(numFactors)
    super.transformSchema(schema)
  }
}
