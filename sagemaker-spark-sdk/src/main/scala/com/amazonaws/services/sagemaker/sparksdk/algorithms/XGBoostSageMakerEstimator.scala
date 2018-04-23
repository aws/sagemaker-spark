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
import com.amazonaws.services.sagemaker.sparksdk.transformation.deserializers.XGBoostCSVRowDeserializer
import com.amazonaws.services.sagemaker.sparksdk.transformation.serializers.LibSVMRequestRowSerializer

private[algorithms] trait XGBoostParams extends Params {

  /* General parameters */
  /** Which booster to use. Can be gbtree, gblinear or dart.
    * The gbtree and dart values use a tree based model while gblinear uses a linear function.
    * Default = gbtree
    */
  val booster : Param[String] = new Param(this, "booster",
    "Which booster to use. Can be gbtree, gblinear or dart. gbtree and dart use tree based model " +
    "while gblinear uses linear function.",
    ParamValidators.inArray(Array("gbtree", "gblinear", "dart")))
  def getBooster: String = $(booster)

  /** Whether in silent mode. Can be 0 or 1.
    * 0 means printing running messages, 1 means silent mode.
    * Default = 0
    */
  val silent : IntParam = new IntParam(this, "silent",
    "Whether in silent mode. Can be 0 or 1. " +
    "0 means print running messages, 1 means silent mode.",
    ParamValidators.inArray(Array(0, 1)))
  def getSilent: Int = $(silent)

  /** Number of parallel threads used to run xgboost. Must be >= 1.
    * Defaults to maximum number of threads available.
    */
  val nThread: IntParam = new IntParam(this, "nthread",
    "Number of parallel threads used to run xgboost. Must be >= 1. ",
    ParamValidators.gtEq(1))
  def getNThread: Int = $(nThread)

  /* Booster Parameters */
  /** Step size shrinkage used in update to prevent overfitting. After each boosting step, we can
    * directly get the weights of new features and eta actually shrinks the feature weights to make
    * the boosting process more conservative. Must be in [0, 1]
    * Default = 0.3
    */
  val eta: DoubleParam = new DoubleParam(this, "eta",
    "Step size shrinkage used in update to prevent overfitting. After each boosting step, " +
    "we can directly get the weights of new features. and eta shrinks the feature " +
    "weights to make the boosting process more conservative. Must be in [0, 1]. ",
    ParamValidators.inRange(0, 1))
  def getEta: Double = $(eta)

  /** Minimum loss reduction required to make a further partition on a leaf node of the tree.
    * The larger, the more conservative the algorithm will be. Must be >= 0.
    * Default = 0
    */
  val gamma: DoubleParam = new DoubleParam(this, "gamma",
    "Minimum loss reduction required to make an additional partition on a leaf node of the tree. " +
    "The larger the value, the more conservative the algorithm will be. Must be >= 0.",
    ParamValidators.gtEq(0))
  def getGamma: Double = $(gamma)

  /** Maximum depth of a tree, increase this value will make the model more complex (likely to be
    * overfitting). 0 indicates no limit, limit is required when grow_policy=depth-wise.
    * Must be >= 0.
    * Default = 6
    */
  val maxDepth: IntParam = new IntParam(this, "max_depth",
    " Maximum depth of a tree, increase this value will make the model more complex (likely to be" +
      " overfitting). 0 indicates no limit, limit is required when grow_policy=depth-wise. " +
      "Must be >= 0. ",
    ParamValidators.gtEq(0))
  def getMaxDepth: Int = $(maxDepth)

  /** Minimum sum of instance weight (hessian) needed in a child. If the tree partition step results
    * in a leaf node with the sum of instance weight less than min_child_weight, then the building
    * process will give up further partitioning. In linear regression mode, this simply corresponds
    * to minimum number of instances needed to be in each node. The larger the algorithm is,
    * the more conservative it will be. Must be >= 0.
    * Default = 1
    */
  val minChildWeight: DoubleParam = new DoubleParam(this, "min_child_weight",
    "Minimum sum of instance weight (hessian) needed in a child. If the tree partition step " +
    "results in a leaf node with the sum of instance weight less than min_child_weight, then " +
    "the building process will give up further partitioning. In linear regression mode, this " +
    "simply corresponds to minimum number of instances needed to be in each node. The larger the " +
    "value, the more conservative the algorithm will be. Must be >= 0.", ParamValidators.gtEq(0))
  def getMinChildWeight: Double = $(minChildWeight)

  /** Maximum delta step allowed for each tree's weight estimation can be. Valid inputs: When a
    * positive integer is used, it helps make the update more conservative. The preferred options
    * is to use it in logistic regression. Set it to 1-10 to help control the update. Must be >= 0.
    * Default = 0
    */
  val maxDeltaStep: DoubleParam = new DoubleParam(this, "max_delta_step",
    "Maximum delta step allowed for each tree's weight estimation to be. If the value is set to" +
    " 0, it means there is no constraint. If it is set to a positive value, it can help make the " +
    "update step more conservative. Usually this parameter is not needed, but it might help " +
    "in logistic regression when the classes are extremely imbalanced. Setting it to value of " +
    "1-10 might help control the update. Must be >= 0.", ParamValidators.gtEq(0))
  def getMaxDeltaStep: Double = $(maxDeltaStep)

  /** Subsample ratio of the training instance. Setting it to 0.5 means that XGBoost randomly
    * collected half of the data instances to grow trees and this will prevent overfitting.
    * Must be in (0, 1].
    * Default = 1
    */
  val subsample: DoubleParam = new DoubleParam(this, "subsample",
    "Subsample ratio of the training instance. Setting it to 0.5 means that XGBoost will " +
    "randomly collect half of the data instances to grow trees and this will prevent overfitting." +
      "Must be in (0, 1]. ",
    ParamValidators.inRange(0, 1, lowerInclusive = false, upperInclusive = true))
  def getSubsample: Double = $(subsample)


  /** Subsample ratio of columns when constructing each tree. Must be in (0, 1]
    * Default = 1
    */
  val colSampleByTree: DoubleParam = new DoubleParam(this, "colsample_bytree",
    "Subsample ratio of columns when constructing each tree. Must be in (0, 1]",
    ParamValidators.inRange(0, 1, lowerInclusive = false, upperInclusive = true))
  def getColSampleByTree: Double = $(colSampleByTree)

  /** Subsample ratio of columns for each split, in each level. Must be in (0, 1].
    * Default = 1
    */
  val colSampleByLevel: DoubleParam = new DoubleParam(this, "colsample_bylevel",
    "Subsample ratio of columns for each split, in each level. Must be in (0, 1].",
    ParamValidators.inRange(0, 1, lowerInclusive = false, upperInclusive = true))
  def getColSampleByLevel: Double = $(colSampleByLevel)

  /** L2 regularization term on weights. Increase this value will make model more conservative.
    * Default = 1
    */
  val lambda: DoubleParam = new DoubleParam(this, "lambda",
    "L2 regularization term on weights, increase this value will make model more conservative.")
  def getLambda: Double = $(lambda)

  /** L1 regularization term on weights. Increase this value will make model more conservative.
    * Default = 0
    */
  val alpha: DoubleParam = new DoubleParam(this, "alpha",
    "L1 regularization term on weights, increase this value will make model more conservative.")
  def getAlpha: Double = $(alpha)

  /** The tree construction algorithm used in XGBoost. Can be auto, exact, approx, hist.
    * Default = "auto"
    */
  val treeMethod: Param[String] = new Param(this, "tree_method",
    "The tree construction algorithm used in XGBoost. Can be auto, exact, approx, hist.",
    ParamValidators.inArray(Array("auto", "exact", "approx", "hist")))
  def getTreeMethod: String = $(treeMethod)

  /** Used only for approximate greedy algorithm. Translates into O(1 / sketch_eps) number of
    * bins. Compared to directly select number of bins, this comes with theoretical guarantee with
    * sketch accuracy. Must be in (0, 1).
    * Default = 0.03
    */
  val sketchEps: DoubleParam = new DoubleParam(this, "sketch_eps",
    "Used only for approximate greedy algorithm. Translates into " +
    "O(1 / sketch_eps) number of bins. Compared to directly select number of bins, " +
    "this comes with theoretical guarantee with sketch accuracy. Must be in (0, 1). ",
    ParamValidators.inRange(0, 1, lowerInclusive = false, upperInclusive = false))
  def getSketchEps: Double = $(sketchEps)

  /** Controls the balance of positive and negative weights. It's useful for unbalanced classes.
    * A typical value to consider: sum(negative cases) / sum(positive cases).
    * Default = 1
    */
  val scalePosWeight: DoubleParam = new DoubleParam(this, "scale_pos_weight",
    "Scale the weight of positive examples by this factor. Useful for unbalanced classes")
  def getScalePosWeight: Double = $(scalePosWeight)

  /** A comma-separated string that defines the sequence of tree updaters to run.
    * This provides a modular way to construct and to modify the trees.
    * Default = "grow_colmaker,prune"
    */
  val updater: Param[String] = new Param(this, "updater",
    "A comma separated string defining the sequence of tree updaters to run, providing a modular " +
    "way to construct and to modify the trees. This is an advanced parameter that " +
    "is usually set automatically, depending on some other parameters", updaterValidator)
  def getUpdater: String = $(updater)

  private def updaterValidator: String => Boolean = {
    (value: String) => value.split(",").map(ud => updaterValues.contains(ud.trim)).reduce(_ && _)
  }

  private val updaterValues = Array("grow_colmaker", "distcol", "grow_histmaker",
    "grow_local_histmaker", "grow_skmaker", "sync", "refresh", "prune")

  /** A parameter of the 'refresh' updater plugin. When set to true, tree leaves and
    * tree node stats are updated. When set to false, only tree node stats are updated.
    * Default = 1
    */
  val refreshLeaf: IntParam = new IntParam(this, "refresh_leaf",
    "A parameter of the 'refresh' updater plugin. When this flag is true, tree leafs as " +
    "well as tree nodes' stats are updated. When it is false, only node stats are updated.",
    ParamValidators.inArray(Array(0, 1)))
  def getRefreshLeaf: Int = $(refreshLeaf)

  /** The type of boosting process to run. Can be default or update.
    * Default = "default"
    */
  val processType: Param[String] = new Param(this, "process_type",
    "The type of boosting process to run. Can be default or update.",
    ParamValidators.inArray(Array("default", "update")))
  def getProcessType: String = $(processType)

  /** Controls the way that new nodes are added to the tree. Can be "depthwise" or "lossguide".
    * Currently supported only if tree_method is set to hist.
    * Default = "depthwise"
    */
  val growPolicy: Param[String] = new Param(this, "grow_policy",
    "Controls the way new nodes are added to the tree. Can be 'depthwise' or 'lossguide'.",
    ParamValidators.inArray(Array("depthwise", "lossguide")))
  def getGrowPolicy: String = $(growPolicy)

  /** Maximum number of nodes to be added. Relevant only if grow_policy = lossguide. Must be >= 0.
    * Default = 0
    */
  val maxLeaves: IntParam = new IntParam(this, "max_leaves",
    "Maximum number of nodes to be added. Only relevant for the 'lossguide' grow policy. " +
      "Must be >= 0. ",
    ParamValidators.gtEq(0))
  def getMaxLeaves: Int = $(maxLeaves)

  /** Maximum number of discrete bins to bucket continuous features. Used only if tree_method=hist.
    * Default = 256
    */
  val maxBin: IntParam = new IntParam(this, "max_bin",
    "Maximum number of discrete bins to bucket continuous features. This is only used if " +
    "'hist' is specified as tree_method. ", ParamValidators.gtEq(1))
  def getMaxBin: Int = $(maxBin)

  /* Dart Booster parameters */
  /** Type of sampling algorithm. Can be "uniform" or "weighted".
    * Default = "uniform"
    */
  val sampleType: Param[String] = new Param(this, "sample_type",
    "Type of sampling algorithm. Can be 'uniform' or 'weighted'. " +
    "\"uniform\": dropped trees are selected uniformly." +
    "\"weighted\": dropped trees are selected in proportion to weight.",
    ParamValidators.inArray(Array("uniform", "weighted")))
  def getSampleType: String = $(sampleType)

  /** Type of normalization algorithm. Can be "tree" or "forest".
    * Default = "tree"
    */
  val normalizeType: Param[String] = new Param(this, "normalize_type",
    "Type of normalization algorithm. Can be 'tree' or 'forest'." +
    "\"tree\": new trees have the same weight of each of dropped trees." +
    "\"forest\": new trees have the same weight of sum of dropped trees (forest).",
    ParamValidators.inArray(Array("tree", "forest")))
  def getNormalizeType: String = $(normalizeType)

  /** Dropout rate (a fraction of previous trees to drop during the dropout). Must be in [0, 1].
    * Default = 0.0
    */
  val rateDrop: DoubleParam = new DoubleParam(this, "rate_drop",
    "dropout rate (a fraction of previous trees to drop during the dropout). Must be in [0, 1]. ",
    ParamValidators.inRange(0.0, 1.0))
  def getRateDrop: Double = $(rateDrop)

  /** Whether to drop at least one tree during the dropout.
    * Default = 0
    */
  val oneDrop: IntParam = new IntParam(this, "one_drop",
    "whether to drop at least one tree during the dropout. ",
    ParamValidators.inArray(Array(0, 1)))
  def getOneDrop: Int = $(oneDrop)

  /** Probability of skipping the dropout procedure during a boosting iteration. Must be in [0, 1].
    * Default: 0
    */
  val skipDrop: DoubleParam = new DoubleParam(this, "skip_drop",
    "Probability of skipping the dropout procedure during a boosting iteration. Must be in [0, 1].",
    ParamValidators.inRange(0.0, 1.0))
  def getSkipDrop: Double = $(skipDrop)

  /* Parameters for linear booster */
  /** L2 regularization term on bias. Must be in [0, 1].
    * Default = 0.0
    */
  val lambdaBias: DoubleParam = new DoubleParam(this, "lambda_bias",
    "L2 regularization term on bias. Must be in [0, 1].", ParamValidators.inRange(0, 1))
  def getLambdaBias: Double = $(lambdaBias)


  /* Parameters for Tweedie Regression */
  /** Parameter that controls the variance of the Tweedie distribution. Must be in (1, 2).
    * Default = 1.5
    */
  val tweedieVariancePower: DoubleParam = new DoubleParam(this, "tweedie_variance_power",
    "parameter that controls the variance of the Tweedie distribution. Must be in (1, 2).",
    ParamValidators.inRange(1, 2, lowerInclusive = false, upperInclusive = false))
  def getTweedieVariancePower: Double = $(tweedieVariancePower)

  /* Learning task parameters */
  /** Specifies the learning task and the corresponding learning objective.
    * Default: "reg:linear"
    */
  val objective: Param[String] = new Param(this, "objective", "Specifies the learning objective." +
    "\"reg:linear\" -- linear regression " +
    "\"reg:logistic\" --logistic regression " +
    "\"binary:logistic\" --logistic regression for binary classification, output is probability " +
    "\"binary:logitraw\" --logistic regression for binary classification, output is score before" +
    " logistic transformation " +
    "\"count:poisson\" --poisson regression for count data, output mean of poisson distribution " +
    "max_delta_step is set to 0.7 by default in poisson regression (used to safeguard " +
    "optimization) " +
    "\"multi:softmax\" --multiclass classification using the softmax objective. " +
    "You also need to set num_class(number of classes)" +
    "\"multi:softprob\" --same as softmax, but output a vector of ndata * nclass, which can be" +
    " further reshaped to ndata, nclass matrix. The result contains predicted probability of each" +
    " data point belonging to each class. " +
    "\"rank:pairwise\" --set XGBoost to do ranking task by minimizing the pairwise loss " +
    "\"reg:gamma\" --gamma regression with log-link. Output is a mean of gamma distribution. " +
    "It might be useful, e.g., for modeling insurance claims severity, or for any outcome " +
    "that might be gamma-distributed" +
    "\"reg:tweedie\" --Tweedie regression with log-link. It might be useful, e.g., for " +
    "modeling total loss in insurance, or for any outcome that might be Tweedie-distributed.",
    ParamValidators.inArray(Array("reg:linear", "reg:logistic", "binary:logistic",
      "binary:logistraw", "count:poisson", "multi:softmax", "multi:softprob",
      "rank:pairwise", "reg:gamma", "reg:tweedie")))
  def getObjective: String = $(objective)

  /**
    * No default. Used for softmax multiclass classification.
    */
  val numClasses: IntParam = new IntParam(this, "num_class",
    "Number of classes", ParamValidators.gtEq(1))
  def getNumClasses: Int = $(numClasses)

  /** The initial prediction score of all instances, global bias.
    * Default = 0.5
    */
  val baseScore: DoubleParam = new DoubleParam(this, "base_score",
    "the initial prediction score of all instances, global bias")
  def getBaseScore: Double = $(baseScore)

  /** Evaluation metrics for validation data. A default metric will be assigned according to the
    * objective (rmse for regression, error for classification, and map for ranking )
    * Default according to objective
    */
  val evalMetric: Param[String] = new Param(this, "eval_metric",
    "Evaluation metrics for validation data. A default metric will be assigned according to " +
    "objective (rmse for regression, and error for classification, mean average " +
    "precision for ranking)",
    ParamValidators.inArray(Array("rmse", "mae", "logloss", "error", "error@t", "merror",
    "mlogloss", "auc", "ndcg", "map", "ndcg@n", "ndcg-", "ndcg@n-", "map-", "map@n-")))
  def getEvalMetric: String = $(evalMetric)

  /** Random number seed.
    * Default = 0
    */
  val seed: IntParam = new IntParam(this, "seed", "Random number seed.")
  def getSeed: Int = $(seed)

  /**
    * Number of rounds for gradient boosting. Must be >= 1. Required.
    */
  val numRound: IntParam = new IntParam(this, "num_round",
    "Number of rounds. Must be >= 1. ", ParamValidators.gtEq(1))
  def getNumRound: Int = $(numRound)

}

object XGBoostSageMakerEstimator {
  val algorithmName = "xgboost"
  val algorithmTag = "1"
  val regionAccountMap = SagerMakerRegionAccountMaps.ApplicationsAccountMap
}

/**
  * A [[SageMakerEstimator]] that runs an XGBoost training job in SageMaker and returns a
  * [[SageMakerModel]] that can be used to transform a DataFrame using
  * the hosted XGBoost model. XGBoost is an open-source distributed gradient boosting
  * library that Amazon SageMaker has adapted to run on Amazon SageMaker.
  *
  * XGBoost trains and infers on LibSVM-formatted data. XGBoostSageMakerEstimator uses
  * Spark's LibSVMFileFormat to write the training DataFrame to S3, and serializes
  * Rows to LibSVM for inference, selecting the column named "features" by default,
  * expected to contain a Vector of Doubles.
  *
  * Inferences made against an Endpoint hosting an XGBoost model contain
  * a "prediction" field appended to the input DataFrame as a column of Doubles, containing
  * the prediction corresponding to the given Vector of features.
  *
  * @see [[https://github.com/dmlc/xgboost]] for more on XGBoost.
  *
  * @param sagemakerRole The SageMaker TrainingJob and Hosting IAM Role. Used by a SageMaker to
  *                      access S3 and ECR resources. SageMaker hosted Endpoints instances
  *                      launched by this Estimator run with this role.
  * @param requestRowSerializer Serializes Spark DataFrame [[Row]]s for transformation by
  *                             Models built from this Estimator.
  * @param responseRowDeserializer Deserializes an Endpoint response into a series of [[Row]]s.
  * @param trainingInputS3DataPath An S3 location to upload SageMaker Training Job input data to.
  * @param trainingOutputS3DataPath An S3 location for SageMaker to store Training Job output
  *                                 data to.
  * @param trainingInstanceType The SageMaker TrainingJob Instance Type to use.
  * @param trainingInstanceCount The number of instances of instanceType to run a SageMaker
  *                              Training Job with.
  * @param trainingInstanceVolumeSizeInGB The EBS volume size in gigabytes of each instance
  * @param trainingProjectedColumns The columns to project from the Dataset being fit before
  *                                 training. If an Optional.empty is passed then no specific
  *                                 projection will occur and all columns will be serialized.
  * @param trainingChannelName The SageMaker Channel name to input serialized Dataset fit input to
  * @param trainingContentType The MIME type of the training data.
  * @param trainingS3DataDistribution The SageMaker Training Job S3 data distribution scheme.
  * @param trainingSparkDataFormat The Spark Data Format name used to serialize the Dataset being
  *                                fit for input to SageMaker.
  * @param trainingSparkDataFormatOptions The Spark Data Format Options used during serialization
  *                                       of the Dataset being fit.
  * @param trainingInputMode The SageMaker Training Job Channel input mode.
  * @param trainingCompressionCodec The type of compression to use when serializing the Dataset
  *                                 being fit for input to SageMaker.
  * @param trainingMaxRuntimeInSeconds A SageMaker Training Job Termination Condition
  *                                    MaxRuntimeInHours.
  * @param trainingKmsKeyId A KMS key ID for the Output Data Source
  * @param modelEnvironmentVariables The environment variables that SageMaker will set on the model
  *                                  container during execution.
  * @param endpointInstanceType The SageMaker Endpoint Confing instance type.
  * @param endpointInitialInstanceCount The SageMaker Endpoint Config minimum number of instances
  *                                     that can be used to host modelImage.
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
  * @param uid The unique identifier of this Estimator. Used to represent this stage in
  *            Spark ML pipelines.
  */
class XGBoostSageMakerEstimator(
           override val sagemakerRole : IAMRoleResource = IAMRoleFromConfig(),
           override val trainingInstanceType : String,
           override val trainingInstanceCount : Int,
           override val endpointInstanceType : String,
           override val endpointInitialInstanceCount : Int,
           override val requestRowSerializer : RequestRowSerializer =
             new LibSVMRequestRowSerializer(),
           override val responseRowDeserializer : ResponseRowDeserializer =
             new XGBoostCSVRowDeserializer(),
           override val trainingInputS3DataPath : S3Resource = S3AutoCreatePath(),
           override val trainingOutputS3DataPath : S3Resource = S3AutoCreatePath(),
           override val trainingInstanceVolumeSizeInGB : Int = 1024,
           override val trainingProjectedColumns : Option[List[String]] = None,
           override val trainingChannelName : String = "train",
           override val trainingContentType: Option[String] = Some("libsvm"),
           override val trainingS3DataDistribution : String =
             S3DataDistribution.ShardedByS3Key.toString,
           override val trainingSparkDataFormat : String = "libsvm",
           override val trainingSparkDataFormatOptions : Map[String, String] =
             Map(),
           override val trainingInputMode : String =
             TrainingInputMode.File.toString,
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
      XGBoostSageMakerEstimator.regionAccountMap,
      XGBoostSageMakerEstimator.algorithmName, XGBoostSageMakerEstimator.algorithmTag),
    modelImage = SageMakerImageURIProvider.getImage(
      region.getOrElse(new DefaultAwsRegionProviderChain().getRegion),
      XGBoostSageMakerEstimator.regionAccountMap,
      XGBoostSageMakerEstimator.algorithmName, XGBoostSageMakerEstimator.algorithmTag),
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
    uid) with XGBoostParams {

  def setBooster(value: String) : this.type = set(booster, value)

  def setSilent(value: Int) : this.type = set(silent, value)

  def setNThread(value: Int) : this.type = set(nThread, value)

  def setEta(value: Double) : this.type = set(eta, value)

  def setGamma(value: Double) : this.type = set(gamma, value)

  def setMaxDepth(value: Int) : this.type = set(maxDepth, value)

  def setMinChildWeight(value: Double) : this.type = set(minChildWeight, value)

  def setMaxDeltaStep(value: Double) : this.type = set(maxDeltaStep, value)

  def setSubsample(value: Double) : this.type = set(subsample, value)

  def setColSampleByTree(value: Double) : this.type = set(colSampleByTree, value)

  def setColSampleByLevel(value: Double) : this.type = set(colSampleByLevel, value)

  def setLambda(value: Double) : this.type = set(lambda, value)

  def setAlpha(value: Double) : this.type = set(alpha, value)

  def setTreeMethod(value: String) : this.type = set(treeMethod, value)

  def setSketchEps(value: Double) : this.type = set(sketchEps, value)

  def setScalePosWeight(value: Double) : this.type = set(scalePosWeight, value)

  def setUpdater(value: String) : this.type = set(updater, value)

  def setRefreshLeaf(value: Int) : this.type = set(refreshLeaf, value)

  def setProcessType(value: String) : this.type = set(processType, value)

  def setGrowPolicy(value: String) : this.type = set(growPolicy, value)

  def setMaxLeaves(value: Int) : this.type = set(maxLeaves, value)

  def setMaxBin(value: Int) : this.type = set(maxBin, value)

  def setSampleType(value: String) : this.type = set(sampleType, value)

  def setNormalizeType(value: String) : this.type = set(normalizeType, value)

  def setRateDrop(value: Double) : this.type = set(rateDrop, value)

  def setOneDrop(value: Int) : this.type = set(oneDrop, value)

  def setSkipDrop(value: Double) : this.type = set(skipDrop, value)

  def setLambdaBias(value: Double) : this.type = set(lambdaBias, value)

  def setTweedieVariancePower(value: Double) : this.type = set(tweedieVariancePower, value)

  def setObjective(value: String) : this.type = set(objective, value)

  def setNumClasses(value: Int) : this.type = set(numClasses, value)

  def setBaseScore(value: Double) : this.type = set(baseScore, value)

  def setEvalMetric(value: String) : this.type = set(evalMetric, value)

  def setSeed(value: Int) : this.type = set(seed, value)

  def setNumRound(value: Int) : this.type = set(numRound, value)

  // Check whether required hyper-parameters are set
  override def transformSchema(schema: StructType): StructType = {
    $(numRound)
    super.transformSchema(schema)
  }
}
