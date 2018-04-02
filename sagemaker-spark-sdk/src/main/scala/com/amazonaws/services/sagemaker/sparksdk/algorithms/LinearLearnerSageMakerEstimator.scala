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
import com.amazonaws.services.sagemaker.sparksdk.transformation.deserializers.{LinearLearnerBinaryClassifierProtobufResponseRowDeserializer, LinearLearnerRegressorProtobufResponseRowDeserializer}
import com.amazonaws.services.sagemaker.sparksdk.transformation.serializers.ProtobufRequestRowSerializer

/**
  * Common params for [[LinearLearnerSageMakerEstimator]] with accessors
  */
private[algorithms] trait BinaryClassifierParams extends LinearLearnerParams {
  /**
    * Pick the model with best criteria from the validation dataset for predictor_type is
    * "binary_classifier". Supported options: "accuracy", "f1", "precision_at_target_recall",
    * "recall_at_target_precision" and "cross_entropy_loss".
    * accuracy: model with highest accuracy
    * f1: model with highest f1 score
    * precision_at_target_recall: model with highest precision at a given recall target
    * recall_at_target_precision: model with highest recall at a given precision target
    * cross_entropy_loss: model with lowest cross entropy loss
    * Default: "accuracy".
    */
  val binaryClassifierModelSelectionCriteria: Param[String] = new Param(this,
    "binary_classifier_model_selection_criteria",
    "Pick the model with best criteria from the validation dataset for predictor_type = " +
      "binary_classifier. " +
      "Supported options: 'accuracy', 'f1', 'precision_at_target_recall'," +
      " 'recall_at_target_precision' and 'cross_entropy_loss'.",
    ParamValidators.inArray(Array(
      "accuracy", "f1", "precision_at_target_recall",
      "recall_at_target_precision", "cross_entropy_loss"))
  )
  def getBinaryClassifierModelSelectionCriteria: String = $(binaryClassifierModelSelectionCriteria)

  /**
    * Applicable if binary_classifier_model_selection_criteria is precision_at_target_recall
    * Ignored otherwise. Must be in range (0, 1).
    * Default: 0.8.
    */
  val targetRecall : DoubleParam = new DoubleParam(this, "target_recall",
    "Applicable if binary_classifier_model_selection_criteria is precision_at_target_recall. " +
      "Ignored otherwise. Must be in range (0, 1).",
    ParamValidators.inRange(0.0, 1.0, false, false))
  def getTargetRecall: Double = $(targetRecall)

  /**
    * Applicable if binary_classifier_model_selection_criteria is recall_at_target_precision
    * Ignored otherwise. Must be in range (0, 1).
    * Default: 0.8.
    */
  val targetPrecision : DoubleParam = new DoubleParam(this, "target_precision",
    "Applicable if binary_classifier_model_selection_criteria is recall_at_target_precision. " +
      "Ignored otherwise. Must be in range (0, 1).",
    ParamValidators.inRange(0.0, 1.0, false, false))
  def getTargetPrecision: Double = $(targetPrecision)

  /**
    * Weight assigned to positive examples when training a binary classifier. The weight of
    * negative examples is fixed at 1. If balanced, then a weight will be selected so that errors
    * in classifying negative vs. positive examples have equal impact on the training loss.
    * If auto, the algorithm will attempt to select the weight that optimizes performance.
    * Must be string "auto", "balanced" or float > 0
    * Default: 1.0.
    */
  val positiveExampleWeightMult : Param[String] = new Param(this, "positive_example_weight_mult",
    "Weight assigned to positive examples when training a binary classifier. The weight of" +
      "negative examples is fixed at 1. If balanced, then a weight will be selected so that" +
      "errors in classifying negative vs. positive examples have equal impact on the training" +
      "loss. If auto, the algorithm will attempt to select the weight that optimizes" +
      "performance. Must be string 'auto', 'balanced' or float > 0",
    inArrayOrAboveParamValidator(Array("auto", "balanced"), 0))
  def getPositiveExampleWeightMult: String = $(positiveExampleWeightMult)
}


private[algorithms] trait LinearLearnerParams extends SageMakerAlgorithmParams {

  /**
    * Max number of passes over the data. Must be > 0.
    * Default: 10.
    */
  val epochs : IntParam = new IntParam(this, "epochs",
    "Max number of passes over the data. Must be > 0.",
    ParamValidators.gt(0))
  def getEpochs: Int = $(epochs)

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
    * Whether model should include bias.
    * Default: "True".
    */
  val useBias : Param[String] = new Param(this, "use_bias",
    "Whether model should include bias. ",
    ParamValidators.inArray(Array("True", "False")))
  def getUseBias: Boolean = parseTrueAndFalse(useBias)

  /**
    * Number of models to train in parallel. Must be > 0 or "auto".
    * If default "auto" is selected, the number of parallel models to train will be decided by
    * the algorithm itself.
    * Default: "auto".
    */
  val numModels : Param[String] = new Param(this, "num_models",
    "Number of models to train in parallel. Must be > 0 or 'auto'",
    autoOrAboveParamValidator(0, false))
  def getNumModels: String = $(numModels)

  /**
    * Number of samples to use from validation dataset for doing model calibration
    * (finding the best threshold). Must be > 0.
    * Default: 10000000.
    */
  val numCalibrationSamples : IntParam = new IntParam(this, "num_calibration_samples",
    "Number of samples to use from validation dataset for doing model calibration" +
      " (finding the best threshold). Must be > 0.",
    ParamValidators.gt(0))
  def getNumCalibrationSamples: Int = $(numCalibrationSamples)

  /**
    * Initialization function for the model weights. Supported options: "uniform" and "normal".
    * uniform: uniformly between (-scale, +scale)
    * normal: normal with mean 0 and sigma
    * Default: "uniform".
    */
  val initMethod : Param[String] = new Param(this, "init_method",
    "Initialization function for the model weights. Supported options: 'uniform' and 'normal'.",
    ParamValidators.inArray(Array("uniform", "normal")))
  def getInitMethod: String = $(initMethod)

  /**
    * Scale for init method uniform. Must be > 0.
    * Default: 0.07.
    */
  val initScale : DoubleParam = new DoubleParam(this, "init_scale",
    "Scale for init method uniform. Must be > 0.",
    ParamValidators.gt(0))
  def getInitScale: Double = $(initScale)

  /**
    * Standard deviation for init method normal. Must be > 0.
    * Default: 0.01.
    */
  val initSigma : DoubleParam = new DoubleParam(this, "init_sigma",
    "Standard deviation for init method normal. Must be > 0.",
    ParamValidators.gt(0))
  def getInitSigma: Double = $(initSigma)

  /**
    * Initial weight for bias.
    * Default: 0.
    */
  val initBias : DoubleParam = new DoubleParam(this, "init_bias",
    "Initial weight for bias. " + "Must be number.")
  def getInitBias: Double = $(initBias)

  /**
    * Which optimizer is to be used. Supported options: "sgd" and "adam".
    * Default: "adam".
    */
  val optimizer : Param[String] = new Param(this, "optimizer", "Which optimizer is to be used. " +
    "Supported options: 'sgd' and 'adam'.",
    ParamValidators.inArray(Array("sgd", "adam")))
  def getOptimizer: String = $(optimizer)

  /**
    * The loss function to apply. Supported options: "logistic", "squared_loss" and "auto".
    * Default: "auto".
    */
  val loss : Param[String] = new Param(this, "loss", "The loss function to apply. " +
    "Supported options: 'logistic', 'squared_loss', 'absolute_loss', 'hinge_loss'," +
    "'eps_insensitive_squared_loss', 'eps_insensitive_absolute_loss', 'quantile_loss'," +
    "'huber_loss' and 'auto'.",
    ParamValidators.inArray(Array("logistic", "squared_loss", "absolute_loss", "hinge_loss",
      "eps_insensitive_squared_loss", "eps_insensitive_absolute_loss", "quantile_loss",
      "huber_loss", "auto")))
  def getLoss: String = $(loss)

  /**
    * The L2 regularization, i.e. the weight decay parameter. Use 0 for no L2 regularization.
    * Must be >= 0.
    * Default: 0.
    */
  val wd : DoubleParam = new DoubleParam(this, "wd",
    "The L2 regularization, i.e. the weight decay parameter. Must be >= 0.",
    ParamValidators.gtEq(0))
  def getWd: Double = $(wd)

  /**
    * The L1 regularization parameter. Use 0 for no L1 regularization. Must be >= 0.
    * Default: 0.
    */
  val l1 : DoubleParam = new DoubleParam(this, "l1",
    "The L1 regularization parameter. Use 0 for no L1 regularization. Must be >= 0.",
    ParamValidators.gtEq(0))
  def getL1: Double = $(l1)

  /**
    * Momentum parameter of sgd optimizer. Must be in range [0, 1).
    * Default: 0.
    */
  val momentum : DoubleParam = new DoubleParam(this, "momentum",
    "Momentum parameter of sgd optimizer. Must be in range [0, 1).",
    ParamValidators.inRange(0.0, 1.0, true, false))
  def getMomentum: Double = $(momentum)

  /**
    * The learning rate. Must be > 0 or "auto".
    * Default: "auto".
    */
  val learningRate : Param[String] = new Param(this, "learning_rate",
    "The learning rate. Must be > 0 or 'auto'",
    autoOrAboveParamValidator(0, false))
  def getLearningRate: String = $(learningRate)

  /**
    * Parameter specific to adam optimizer. Exponential decay rate for first moment estimates.
    * Ignored when optimizer is not adam. Must be in range [0, 1).
    * Default: 0.9.
    */
  val beta1 : DoubleParam = new DoubleParam(this, "beta_1",
    "Parameter specific to adam optimizer. Exponential decay rate for first moment estimates. " +
      "Ignored when optimizer is not adam. Must be in range [0, 1).",
    ParamValidators.inRange(0.0, 1.0, true, false))
  def getBeta1: Double = $(beta1)

  /**
    * Parameter specific to adam optimizer. Exponential decay rate for second moment estimates.
    * Ignored when optimizer is not adam. Must be in range [0, 1).
    * Default: 0.999.
    */
  val beta2 : DoubleParam = new DoubleParam(this, "beta_2",
    "Parameter specific to adam optimizer. exponential decay rate for second moment estimates. " +
      "Ignored when optimizer is not adam. Must be in range [0, 1).",
    ParamValidators.inRange(0.0, 1.0, true, false))
  def getBeta2: Double = $(beta2)

  /**
    * Learning rate bias multiplier.
    * The actual learning rate for the bias is learning rate times bias_lr_mult. Must be > 0.
    * Default: 10.
    */
  val biasLrMult : DoubleParam = new DoubleParam(this, "bias_lr_mult",
    "Learning rate bias multiplier. " +
      "The actual learning rate for the bias is learning rate times bias_lr_mult. " +
      "Must be > 0.", ParamValidators.gt(0))
  def getBiasLrMult: Double = $(biasLrMult)

  /**
    * Weight decay parameter multiplier.
    * The actual L2 regularization weight for the bias is wd times bias_wd_mult. Must be >= 0.
    * Default: 0.
    */
  val biasWdMult : DoubleParam = new DoubleParam(this, "bias_wd_mult",
    "Weight decay parameter multiplier. " +
      "The actual L2 regularization weight for the bias is wd times bias_wd_mult. " +
      "Must be >= 0.", ParamValidators.gtEq(0))
  def getBiasWdMult: Double = $(biasWdMult)

  /**
    * Whether to use a scheduler for the learning rate.
    * Default: True
    */
  val useLrScheduler : Param[String] = new Param(this, "use_lr_scheduler",
    "Whether to use a scheduler for the learning rate. ",
    ParamValidators.inArray(Array("True", "False")))
  def getUseLrScheduler: Boolean = parseTrueAndFalse(useLrScheduler)

  /**
    * Parameter specific to lr_scheduler. Ignored otherwise.
    * The number of steps between decreases of the learning rate. Must be > 0.
    * Default: 100.
    */
  val lrSchedulerStep : IntParam = new IntParam(this, "lr_scheduler_step",
    "Parameter specific to lr_scheduler. Ignored otherwise." +
      "The number of steps between decreases of the learning rate. " +
      "Must be > 0.", ParamValidators.gt(0))
  def getLrSchedulerStep: Int = $(lrSchedulerStep)

  /**
    * Parameter specific to lr_scheduler. Ignored otherwise.
    * Every lr_scheduler_step the learning rate will decrease by this quantity. Must be in (0, 1).
    * Default: 0.99.
    */
  val lrSchedulerFactor : DoubleParam = new DoubleParam(this, "lr_scheduler_factor",
    "Parameter specific to lr_scheduler. Ignored otherwise." +
      "Every lr_scheduler_step the learning rate will decrease by this quantity. " +
      "Must be in (0, 1).", ParamValidators.inRange(0, 1, false, false))
  def getLrSchedulerFactor: Double = $(lrSchedulerFactor)

  /**
    * Parameter specific to lr_scheduler. Ignored otherwise.
    * The learning rate will never decrease to a value lower than lr_scheduler_minimum_lr.
    * Must be > 0.
    * Default: 1e-5.
    */
  val lrSchedulerMinimumLr : DoubleParam = new DoubleParam(this, "lr_scheduler_minimum_lr",
    "Parameter specific to lr_scheduler. Ignored otherwise." +
      "The learning rate will never decrease to a value lower than lr_scheduler_minimum_lr. " +
      "Must be > 0.", ParamValidators.gt(0))
  def getLrSchedulerMinimumLr: Double = $(lrSchedulerMinimumLr)

  /**
    * Whether to normalize the features before training to have std_dev of 1.
    * Default: True
    */
  val normalizeData : Param[String] = new Param(this, "normalize_data",
    "Whether to normalize the features before training to have std_dev of 1. ",
    ParamValidators.inArray(Array("True", "False")))
  def getNormalizeData: Boolean = parseTrueAndFalse(normalizeData)

  /**
    * Whether regression label is normalized. Ignored in classification.
    * Default: "auto"
    */
  val normalizeLabel : Param[String] = new Param(this, "normalize_label",
    "Whether regression label is normalized. If set for classification, it will be ignored.",
    ParamValidators.inArray(Array("True", "False")))
  def getNormalizeLabel: Boolean = parseTrueAndFalse(normalizeLabel)

  /**
    * Whether to unbias the features before training so that mean is 0.
    * By default data is unbiased if use_bias is set to true.
    * Default: "auto"
    */
  val unbiasData : Param[String] = new Param(this, "unbias_data",
    "Whether to unbias the features before training so that mean is 0. " +
      "By default data is unbiased if use_bias is set to true.",
    ParamValidators.inArray(Array("True", "False")))
  def getUnbiasData: Boolean = parseTrueAndFalse(unbiasData)

  /**
    * Whether to unbias the labels before training so that mean is 0.
    * Only done for regrssion if use_bias is true. Otherwise will be ignored.
    * Default: "auto"
    */
  val unbiasLabel : Param[String] = new Param(this, "unbias_label",
    "Whether to unbias the labels before training so that mean is 0. " +
      "Only done for regrssion if use_bias is true. Otherwise will be ignored.",
    ParamValidators.inArray(Array("True", "False")))
  def getUnbiasLabel: Boolean = parseTrueAndFalse(unbiasLabel)

  /**
    * Number of data points to use for calcuating the normalizing / unbiasing terms. Must be > 0.
    * Default: 10000.
    */
  val numPointForScaler : IntParam = new IntParam(this, "num_point_for_scaler",
    "Number of data points to use for calcuating the normalizing / unbiasing terms. " +
      "Must be > 0.", ParamValidators.gt(0))
  def getNumPointForScaler: Int = $(numPointForScaler)

  /**
    * The number of epochs to wait before ending training if no improvement is made in the relevant
    * metric. The metric is the binary_classifier_model_selection_criteria if provided, otherwise
    * the metric is the same as loss. The metric is evaluated on the validation data. If no
    * validation data is provided, the metric is always the same as loss and is evaluated on the
    * training data. To disable early stopping, set early_stopping_patience to a value larger than
    * epochs. Must be > 0.
    * Default: 3.
    */
  val earlyStoppingPatience : IntParam = new IntParam(this, "early_stopping_patience",
    "The number of epochs to wait before ending training if no improvement is made in the" +
      "relevant metric. The metric is the binary_classifier_model_selection_criteria if" +
      "provided,otherwise the metric is the same as loss. The metric is evaluated on the" +
      "validation data. If no validation data is provided, the metric is always the same as loss" +
      "and is evaluated on the training data. To disable early stopping, set" +
      "early_stopping_patience to a value larger than epochs. Must be > 0.", ParamValidators.gt(0))
  def getEarlyStoppingPatience: Int = $(earlyStoppingPatience)

  /**
    * Relative tolerance to measure an improvement in loss. If the ratio of the improvement in loss
    * divided by the previous best loss is smaller than this value, early stopping will consider
    * the improvement to be zero. Must be > 0.
    * Default: 0.001.
    */
  val earlyStoppingTolerance : DoubleParam = new DoubleParam(this, "early_stopping_tolerance",
    "Relative tolerance to measure an improvement in loss. If the ratio of the improvement in" +
      "loss divided by the previous best loss is smaller than this value, early stopping will" +
      "consider the improvement to be zero. Must be > 0.", ParamValidators.gt(0))
  def getEarlyStoppingTolerance: Double = $(earlyStoppingTolerance)

  /**
    * Margin for hinge_loss. Must be > 0.
    * Default: 1.0.
    */
  val margin : DoubleParam = new DoubleParam(this, "margin",
    "Margin for hinge_loss. Must be > 0.", ParamValidators.gt(0))
  def getMargin: Double = $(margin)

  /**
    * Quantile for quantile loss. For quantile q, the model will attempt to produce predictions
    * such that true_label < prediction with probability q. Must be in (0, 1).
    * Default: 0.5.
    */
  val quantile : DoubleParam = new DoubleParam(this, "quantile",
    "Quantile for quantile loss. For quantile q, the model will attempt to produce predictions" +
      "such that true_label < prediction with probability q. " +
      "Must be in (0, 1).", ParamValidators.inRange(0, 1, false, false))
  def getQuantile: Double = $(quantile)

  /**
    * Parameter for epsilon insensitive loss type. During training and metric evaluation,
    * any error smaller than this is considered to be zero. Must be > 0.
    * Default: 0.01.
    */
  val lossInsensitivity : DoubleParam = new DoubleParam(this, "loss_insensitivity",
    "Parameter for epsilon insensitive loss type. During training and metric evaluation," +
      "any error smaller than this is considered to be zero. Must be > 0.", ParamValidators.gt(0))
  def getLossInsensitivity: Double = $(lossInsensitivity)

  /**
    * Parameter for Huber loss. During training and metric evaluation, compute L2 loss for errors
    * smaller than delta and L1 loss for errors larger than delta. Must be > 0.
    * Default: 1.0.
    */
  val huberDelta : DoubleParam = new DoubleParam(this, "huber_delta",
    "Parameter for Huber loss. During training and metric evaluation, compute L2 loss for" +
      "errors smaller than delta and L1 loss for errors larger than delta." +
      "Must be > 0.", ParamValidators.gt(0))
  def getHuberDelta: Double = $(huberDelta)
}

object LinearLearnerSageMakerEstimator {
  val algorithmName = "linear-learner"
  val algorithmTag = "1"
  val regionAccountMap = SagerMakerRegionAccountMaps.AlgorithmsAccountMap
}

/**
  * A [[SageMakerEstimator]] that runs a Linear Learner training job in "binary classifier" mode
  * in SageMaker and returns a [[SageMakerModel]] that can be used to transform a DataFrame using
  * the hosted Linear Learner model. The Linear Learner Binary Classifier is useful for classifying
  * examples into one of two classes.
  *
  * Amazon SageMaker Linear Learner trains on RecordIO-encoded Amazon Record protobuf data.
  * SageMaker Spark writes a DataFrame to S3 by selecting a column of Vectors named "features"
  * and, if present, a column of Doubles named "label". These names are configurable by passing
  * a map with entries in trainingSparkDataFormatOptions with key "labelColumnName" or
  * "featuresColumnName", with values corresponding to the desired label and features columns.
  *
  * Inferences made against an Endpoint hosting a Linear Learner Binary classifier model contain
  * a "score" field and a "predicted_label" field, both appended to the input DataFrame as
  * Doubles.
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
class LinearLearnerBinaryClassifier(
           override val sagemakerRole : IAMRoleResource = IAMRoleFromConfig(),
           override val trainingInstanceType : String,
           override val trainingInstanceCount : Int,
           override val endpointInstanceType : String,
           override val endpointInitialInstanceCount : Int,
           override val requestRowSerializer : RequestRowSerializer =
           new ProtobufRequestRowSerializer(),
           override val responseRowDeserializer : ResponseRowDeserializer =
           new LinearLearnerBinaryClassifierProtobufResponseRowDeserializer(),
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
  extends LinearLearnerSageMakerEstimator(
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
    uid) with BinaryClassifierParams {

  setDefault(predictorType -> "binary_classifier")

  def setBinaryClassifierModelSelectionCriteria(value: String): this.type =
    set(binaryClassifierModelSelectionCriteria, value)

  def setTargetRecall(value: Double): this.type = set(targetRecall, value)

  def setTargetPrecision(value: Double): this.type = set(targetPrecision, value)

  def setPositiveExampleWeightMult(value: String): this.type = set(positiveExampleWeightMult, value)

  def setPositiveExampleWeightMult(value: Double): this.type = set(positiveExampleWeightMult,
                                                                   value.toString())
}


/**
  * A [[SageMakerEstimator]] that runs a Linear Learner training job in "regressor" mode in
  * SageMaker and returns a [[SageMakerModel]] that can be used to transform a DataFrame using
  * the hosted Linear Learner model. The Linear Learner Regressor is useful for predicting
  * a real-valued label from training examples.
  *
  * Amazon SageMaker Linear Learner trains on RecordIO-encoded Amazon Record protobuf data.
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
  * Inferences made against an Endpoint hosting a Linear Learner Regressor model contain
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
class LinearLearnerRegressor(
                override val sagemakerRole : IAMRoleResource = IAMRoleFromConfig(),
                override val trainingInstanceType : String,
                override val trainingInstanceCount : Int,
                override val endpointInstanceType : String,
                override val endpointInitialInstanceCount : Int,
                override val requestRowSerializer : RequestRowSerializer =
                  new ProtobufRequestRowSerializer(),
                override val responseRowDeserializer : ResponseRowDeserializer =
                  new LinearLearnerRegressorProtobufResponseRowDeserializer(),
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
  extends LinearLearnerSageMakerEstimator(
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
    uid) with LinearLearnerParams {
      setDefault(predictorType -> "regressor")
  }

/**
  * A [[SageMakerEstimator]] that runs a Linear Learner training job in SageMaker and returns
  * a [[SageMakerModel]] that can be used to transform a DataFrame using
  * the hosted Linear Learner model. This algorithm can be run in either as a Linear Regressor
  * (see [[LinearLearnerRegressor]]) or as a Binary Classifier
  * (see [[LinearLearnerBinaryClassifier]]).
  *
  * Amazon SageMaker Linear Learner trains on RecordIO-encoded Amazon Record protobuf data.
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
private[algorithms] class LinearLearnerSageMakerEstimator(
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
      LinearLearnerSageMakerEstimator.regionAccountMap,
      LinearLearnerSageMakerEstimator.algorithmName,
      LinearLearnerSageMakerEstimator.algorithmTag),
    modelImage = SageMakerImageURIProvider.getImage(
      region.getOrElse(new DefaultAwsRegionProviderChain().getRegion),
      LinearLearnerSageMakerEstimator.regionAccountMap,
      LinearLearnerSageMakerEstimator.algorithmName,
      LinearLearnerSageMakerEstimator.algorithmTag),
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
    uid) with LinearLearnerParams {

  setDefault(miniBatchSize -> 1000)

  def setFeatureDim(value: Int): this.type = set(featureDim, value)

  def setMiniBatchSize(value: Int): this.type = set(miniBatchSize, value)

  def setEpochs(value: Int): this.type = set(epochs, value)

  def setUseBias(value: Boolean): this.type = set(useBias, value match {
    case true => "True"
    case false => "False"
  })

  def setNumModels(value: Int): this.type = set(numModels, value.toString)

  def setNumModels(value: String): this.type = set(numModels, value)

  def setNumCalibrationSamples(value: Int): this.type = set(numCalibrationSamples, value)

  def setInitMethod(value: String): this.type = set(initMethod, value)

  def setInitScale(value: Double): this.type = set(initScale, value)

  def setInitSigma(value: Double): this.type = set(initSigma, value)

  def setInitBias(value: Double): this.type = set(initBias, value)

  def setOptimizer(value: String): this.type = set(optimizer, value)

  def setLoss(value: String): this.type = set(loss, value)

  def setWd(value: Double): this.type = set(wd, value)

  def setL1(value: Double): this.type = set(l1, value)

  def setMomentum(value: Double): this.type = set(momentum, value)

  def setLearningRate(value: Double): this.type = set(learningRate, value.toString)

  def setLearningRate(value: String): this.type = set(learningRate, value)

  def setBeta1(value: Double): this.type = set(beta1, value)

  def setBeta2(value: Double): this.type = set(beta2, value)

  def setBiasLrMult(value: Double): this.type = set(biasLrMult, value)

  def setBiasWdMult(value: Double): this.type = set(biasWdMult, value)

  def setUseLrScheduler(value: Boolean): this.type = set(useLrScheduler, value match {
    case true => "True"
    case false => "False"
  })

  def setLrSchedulerStep(value: Int): this.type = set(lrSchedulerStep, value)

  def setLrSchedulerFactor(value: Double): this.type = set(lrSchedulerFactor, value)

  def setLrSchedulerMinimumLr(value: Double): this.type = set(lrSchedulerMinimumLr, value)

  def setNormalizeData(value: Boolean): this.type = set(normalizeData, value match {
    case true => "True"
    case false => "False"
  })

  def setNormalizeLabel(value: Boolean): this.type = set(normalizeLabel, value match {
    case true => "True"
    case false => "False"
  })

  def setUnbiasData(value: Boolean): this.type = set(unbiasData, value match {
    case true => "True"
    case false => "False"
  })

  def setUnbiasLabel(value: Boolean): this.type = set(unbiasLabel, value match {
    case true => "True"
    case false => "False"
  })

  def setNumPointForScaler(value: Int): this.type = set(numPointForScaler, value)

  def setEarlyStoppingPatience(value: Int): this.type = set(earlyStoppingPatience, value)

  def setEarlyStoppingTolerance(value: Double): this.type = set(earlyStoppingTolerance, value)

  def setMargin(value: Double): this.type = set(margin, value)

  def setQuantile(value: Double): this.type = set(quantile, value)

  def setLossInsensitivity(value: Double): this.type = set(lossInsensitivity, value)

  def setHuberDelta(value: Double): this.type = set(huberDelta, value)


  // Check whether required hyper-parameters are set
  override def transformSchema(schema: StructType): StructType = {
    $(featureDim)
    $(predictorType)
    $(miniBatchSize)
    super.transformSchema(schema)
  }

}
