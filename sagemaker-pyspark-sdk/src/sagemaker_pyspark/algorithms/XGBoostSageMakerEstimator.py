# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#   http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from pyspark.ml.param import Params, Param, TypeConverters
from pyspark.ml.util import Identifiable

from sagemaker_pyspark import (SageMakerEstimatorBase, S3AutoCreatePath, Option, IAMRoleFromConfig,
                               EndpointCreationPolicy, SageMakerClients, RandomNamePolicyFactory)
from sagemaker_pyspark.transformation.serializers import LibSVMRequestRowSerializer
from sagemaker_pyspark.transformation.deserializers import XGBoostCSVRowDeserializer


class XGBoostSageMakerEstimator(SageMakerEstimatorBase):
    """
    A :class:`~sagemaker_pyspark.SageMakerEstimator` that runs an XGBoost training job in
    Amazon SageMaker and returns a :class:`~sagemaker_pyspark.SageMakerModel` that can be used to
    transform a DataFrame using he hosted XGBoost model.  XGBoost is an open-source distributed
    gradient boosting library that Amazon SageMaker has adapted to run on Amazon SageMaker.

    XGBoost trains and infers on LibSVM-formatted data. XGBoostSageMakerEstimator uses Spark's
    LibSVMFileFormat to write the training DataFrame to S3, and serializes Rows to LibSVM for
    inference, selecting the column named "features" by default, expected to contain a Vector of
    Doubles.

    Inferences made against an Endpoint hosting an XGBoost model contain a "prediction" field
    appended to the input DataFrame as a column of Doubles, containing the prediction corresponding
    to the given Vector of features.

    See `XGBoost github <https://github.com/dmlc/xgboost>`__ for more on XGBoost

    Args:
        sageMakerRole (IAMRole): The SageMaker TrainingJob and Hosting IAM Role. Used by
            SageMaker to access S3 and ECR Resources. SageMaker hosted Endpoint instances
            launched by this Estimator run with this role.
        trainingInstanceType (str): The SageMaker TrainingJob Instance Type to use.
        trainingInstanceCount (int): The number of instances of instanceType to run an
            SageMaker Training Job with.
        endpointInstanceType (str): The SageMaker Endpoint Config instance type.
        endpointInitialInstanceCount (int): The SageMaker Endpoint Config minimum number of
            instances that can be used to host modelImage.
        requestRowSerializer (RequestRowSerializer): Serializes Spark DataFrame Rows for
            transformation by Models built from this Estimator.
        responseRowDeserializer (ResponseRowDeserializer): Deserializes an Endpoint response into a
            series of Rows.
        trainingInputS3DataPath (S3Resource): An S3 location to upload SageMaker Training Job input
            data to.
        trainingOutputS3DataPath (S3Resource): An S3 location for SageMaker to store Training Job
            output data to.
        trainingInstanceVolumeSizeInGB (int): The EBS volume size in gigabytes of each instance.
        trainingProjectedColumns (List): The columns to project from the Dataset being fit before
            training. If an Optional.empty is passed then no specific projection will occur and
            all columns will be serialized.
        trainingChannelName (str): The SageMaker Channel name to input serialized Dataset fit
            input to.
        trainingContentType (str): The MIME type of the training data.
        trainingS3DataDistribution (str): The SageMaker Training Job S3 data distribution scheme.
        trainingSparkDataFormat (str): The Spark Data Format name used to serialize the Dataset
            being fit for input to SageMaker.
        trainingSparkDataFormatOptions (dict): The Spark Data Format Options used during
            serialization of the Dataset being fit.
        trainingInputMode (str): The SageMaker Training Job Channel input mode.
        trainingCompressionCodec (str): The type of compression to use when serializing the
            Dataset being fit for input to SageMaker.
        trainingMaxRuntimeInSeconds (int): A SageMaker Training Job Termination Condition
            MaxRuntimeInHours.
        trainingKmsKeyId (str): A KMS key ID for the Output Data Source.
        modelEnvironmentVariables (dict): The environment variables that SageMaker will set on the
            model container during execution.
        endpointCreationPolicy (EndpointCreationPolicy): Defines how a SageMaker Endpoint
            referenced by a SageMakerModel is created.
        sagemakerClient (AmazonSageMaker) Amazon SageMaker client. Used to send CreateTrainingJob,
            CreateModel, and CreateEndpoint requests.
        region (str): The region in which to run the algorithm. If not specified, gets the region
            from the DefaultAwsRegionProviderChain.
        s3Client (AmazonS3): Used to create a bucket for staging SageMaker Training Job
            input and/or output if either are set to S3AutoCreatePath.
        stsClient (AmazonSTS): Used to resolve the account number when creating staging
            input / output buckets.
        modelPrependInputRowsToTransformationRows (bool): Whether the transformation result on
            Models built by this Estimator should also include the input Rows. If true,
            each output Row is formed by a concatenation of the input Row with the corresponding
            Row produced by SageMaker Endpoint invocation, produced by responseRowDeserializer.
            If false, each output Row is just taken from responseRowDeserializer.
        deleteStagingDataAfterTraining (bool): Whether to remove the training data on s3 after
            training is complete or failed.
        namePolicyFactory (NamePolicyFactory): The NamePolicyFactory to use when naming SageMaker
            entities created during fit.
        uid (str): The unique identifier of this Estimator. Used to represent this stage in Spark
            ML pipelines.

       """
    _wrapped_class = \
        "com.amazonaws.services.sagemaker.sparksdk.algorithms.XGBoostSageMakerEstimator"

    booster = Param(
        Params._dummy(), "booster",
        "Which booster to use. Can be 'gbtree', 'gblinear' or 'dart'. "
        "gbtree and dart use tree based model while gblinear uses linear function.",
        typeConverter=TypeConverters.toString)

    silent = Param(
        Params._dummy(), "silent",
        "Whether in silent mode."
        "0 means print running messages, 1 means silent mode.",
        typeConverter=TypeConverters.toInt)

    nthread = Param(
        Params._dummy(), "nthread",
        "Number of parallel threads used to run xgboot. Must be >= 1.",
        typeConverter=TypeConverters.toInt)

    eta = Param(
        Params._dummy(), "eta",
        "Step size shrinkage used in update to prevent overfitting. After each boosting step, "
        "we can directly get the weights of new features. and eta shrinks the feature weights "
        "to make the boosting process more conservative. Must be in [0, 1].",
        typeConverter=TypeConverters.toFloat)

    gamma = Param(
        Params._dummy(), "gamma",
        "Minimum loss reduction required to make an additional partition on a leaf node"
        " of the tree. The larger the value, the more conservative the algorithm will be."
        "Must be >= 0.",
        typeConverter=TypeConverters.toFloat)

    max_depth = Param(
        Params._dummy(), "max_depth",
        "Maximum depth of a tree. Increasing this value makes the model more complex and "
        "likely to be overfitted. 0 indicates no limit. A limit is required when"
        "grow_policy=depth-wise. Must be >= 0. Default value is 6",
        typeConverter=TypeConverters.toInt)

    min_child_weight = Param(
        Params._dummy(), "min_child_weight",
        "Minimum sum of instance weight (hessian) needed in a child. If the tree partition step "
        "results in a leaf node with the sum of instance weight less than min_child_weight, then "
        "the building process will give up further partitioning. In linear regression mode, "
        "this simply corresponds to minimum number of instances needed to be in each node. "
        "The larger the value, the more conservative the algorithm will be. Must be >= 0.",
        typeConverter=TypeConverters.toFloat)

    max_delta_step = Param(
        Params._dummy(), "max_delta_step",
        "Maximum delta step we allow each tree's weight estimation to be. "
        "If the value is set to 0, it means there is no constraint. If it is set to a positive "
        "value, it can help make the update step more conservative. Usually this parameter is "
        "not needed, but it might help in logistic regression when the classes are extremely"
        " imbalanced. Setting it to value of 1-10 might help control the update. Must be >= 0.",
        typeConverter=TypeConverters.toFloat)

    subsample = Param(
        Params._dummy(), "subsample",
        "Subsample ratio of the training instance. Setting it to 0.5 means that XGBoost will "
        "randomly collect half of the data instances to grow trees and this will "
        "prevent overfitting. Must be (0, 1].",
        typeConverter=TypeConverters.toFloat)

    colsample_bytree = Param(
        Params._dummy(), "colsample_bytree",
        "Subsample ratio of columns when constructing each tree. Must be in (0, 1].",
        typeConverter=TypeConverters.toFloat)

    colsample_bylevel = Param(
        Params._dummy(), "colsample_bylevel",
        "Subsample ratio of columns for each split, in each level. Must be in (0, 1].",
        typeConverter=TypeConverters.toFloat)

    _lambda = Param(
        Params._dummy(), "lambda",
        "L2 regularization term on weights, increase this value"
        " will make model more conservative.",
        typeConverter=TypeConverters.toFloat)

    alpha = Param(
        Params._dummy(), "alpha",
        "L1 regularization term on weights, increase this value "
        "will make model more conservative.",
        typeConverter=TypeConverters.toFloat)

    tree_method = Param(
        Params._dummy(), "tree_method",
        "The tree construction algorithm used in XGBoost. Can be "
        "'auto', 'exact', 'approx' or 'hist'",
        typeConverter=TypeConverters.toString)

    sketch_eps = Param(
        Params._dummy(), "sketch_eps",
        "Used only for approximate greedy algorithm. This translates into O(1 / sketch_eps) number"
        "of bins. Compared to directly select number of bins, this comes with theoretical guarantee"
        "with sketch accuracy."
        "Must be in (0, 1).",
        typeConverter=TypeConverters.toFloat)

    scale_pos_weight = Param(
        Params._dummy(), "scale_pos_weight",
        "Controls the balance of positive and negative weights. It's useful for unbalanced classes."
        "A typical value to consider: sum(negative cases) / sum(positive cases).",
        typeConverter=TypeConverters.toFloat)

    updater = Param(
        Params._dummy(), "updater",
        "A comma separated string defining the sequence of tree updaters to run, "
        "providing a modular way to construct and to modify the trees. "
        "This is an advanced parameter that is usually set automatically, "
        "depending on some other parameters. Can be "
        "'grow_colmaker', 'distcol', 'grow_histmaker', 'grow_local_histmaker',"
        "'grow_skmaker', 'sync', 'refresh', 'prune'.",
        typeConverter=TypeConverters.toString)

    refresh_leaf = Param(
        Params._dummy(), "refresh_leaf",
        "This is a parameter of the 'refresh' updater plugin. When set to true, tree leaves and"
        "tree node stats are updated. When set to false, only tree node stats are updated.",
        typeConverter=TypeConverters.toInt)

    process_type = Param(
        Params._dummy(), "process_type",
        "The type of boosting process to run. Can be 'default', 'update'",
        typeConverter=TypeConverters.toString)

    grow_policy = Param(
        Params._dummy(), "grow_policy",
        "Controls the way that new nodes are added to the tree. Currently supported"
        "only if tree_method is set to hist. Can be 'depthwise', 'lossguide'",
        typeConverter=TypeConverters.toString)

    max_leaves = Param(
        Params._dummy(), "max_leaves",
        "Maximum number of nodes to be added. Relevant only if grow_policy = lossguide.",
        typeConverter=TypeConverters.toInt)

    max_bin = Param(
        Params._dummy(), "max_bin",
        "Maximum number of discrete bins to bucket continuous features."
        "Used only if tree_method = hist.",
        typeConverter=TypeConverters.toInt)

    sample_type = Param(
        Params._dummy(), "sample_type",
        "Type of sampling algorithm. Can be 'uniform' or 'weighted'."
        "'uniform': dropped trees are selected uniformly."
        "'weighted': dropped trees are selected in proportion to weight.",
        typeConverter=TypeConverters.toString)

    normalize_type = Param(
        Params._dummy(), "normalize_type",
        "type of normalization algorithm. Can be 'tree' or 'forest'"
        "'tree': new trees have the same weight of each of dropped trees."
        "'forest': new trees have the same weight of sum of dropped trees (forest).",
        typeConverter=TypeConverters.toString)

    rate_drop = Param(
        Params._dummy(), "rate_drop",
        "dropout rate (a fraction of previous trees to drop during the dropout). "
        "Must be in [0.0, 1.0]",
        typeConverter=TypeConverters.toFloat)

    one_drop = Param(
        Params._dummy(), "one_drop",
        "When this flag is enabled, at least one tree is always dropped during the dropout.",
        typeConverter=TypeConverters.toInt)

    skip_drop = Param(
        Params._dummy(), "skip_drop",
        "Probability of skipping the dropout procedure during a boosting iteration."
        "Must be in [0.0, 1.0]",
        typeConverter=TypeConverters.toFloat)

    lambda_bias = Param(
        Params._dummy(), "lambda_bias",
        "L2 regularization term on bias. Must be in [0, 1].",
        typeConverter=TypeConverters.toFloat)

    tweedie_variance_power = Param(
        Params._dummy(), "tweedie_variance_power",
        "parameter that controls the variance of the Tweedie distribution. Must be in (1.0, 2.0).",
        typeConverter=TypeConverters.toFloat)

    objective = Param(
        Params._dummy(), "objective",
        "Specifies the learning objective."
        "\"reg:logistic\" --logistic regression "
        "\"binary:logistic\" --logistic regression for binary classification, "
        "output is probability "
        "\"binary:logitraw\" --logistic regression for binary classification, output is"
        " score before logistic transformation "
        "\"count:poisson\" --poisson regression for count data, output mean of poisson"
        " distribution max_delta_step is set to 0.7 by default in poisson regression "
        "(used to safeguard optimization) "
        "\"multi:softmax\" --multiclass classification using the softmax objective. "
        "You also need to set num_class(number of classes)"
        "\"multi:softprob\" --same as softmax, but output a vector of ndata * nclass, "
        "which can be further reshaped to ndata, nclass matrix. "
        "The result contains predicted probability of each data point belonging to each class. "
        "\"rank:pairwise\" --set XGBoost to do ranking task by minimizing the pairwise loss "
        "\"reg:gamma\" --gamma regression with log-link. Output is a mean of gamma distribution. "
        "It might be useful, e.g., for modeling insurance claims severity, or for any outcome "
        "that might be gamma-distributed"
        "\"reg:tweedie\" --Tweedie regression with log-link. It might be useful, e.g., for "
        "modeling total loss in insurance, or for any outcome that might be"
        " Tweedie-distributed.",
        typeConverter=TypeConverters.toString)

    num_class = Param(
        Params._dummy(), "num_class",
        "Number of classes. >= 1",
        typeConverter=TypeConverters.toInt)

    base_score = Param(
        Params._dummy(), "base_score",
        "the initial prediction score of all instances, global bias. Value range: [0.0, 1.0]",
        typeConverter=TypeConverters.toFloat)

    eval_metric = Param(
        Params._dummy(), "eval_metric",
        "Evaluation metrics for validation data. A default metric will be assigned according to"
        " objective (rmse for regression, and error for classification, mean average "
        "precision for ranking). Values: 'rmse', 'mae', 'logloss', 'error', 'error@t', 'merror',"
        "'mlogloss', 'auc', 'ndcg', 'ndcg@n', 'ndcg@n-', 'map-', 'map@n-'.",
        typeConverter=TypeConverters.toString)

    seed = Param(
        Params._dummy(), "seed",
        "Random number seed",
        typeConverter=TypeConverters.toInt)

    num_round = Param(
        Params._dummy(), "num_round",
        "The number of rounds to run the training. Must be >= 1",
        typeConverter=TypeConverters.toInt)

    def __init__(self,
                 trainingInstanceType,
                 trainingInstanceCount,
                 endpointInstanceType,
                 endpointInitialInstanceCount,
                 sagemakerRole=IAMRoleFromConfig(),
                 requestRowSerializer=LibSVMRequestRowSerializer(),
                 responseRowDeserializer=XGBoostCSVRowDeserializer(),
                 trainingInputS3DataPath=S3AutoCreatePath(),
                 trainingOutputS3DataPath=S3AutoCreatePath(),
                 trainingInstanceVolumeSizeInGB=1024,
                 trainingProjectedColumns=None,
                 trainingChannelName="train",
                 trainingContentType=None,
                 trainingS3DataDistribution="ShardedByS3Key",
                 trainingSparkDataFormat="libsvm",
                 trainingSparkDataFormatOptions=None,
                 trainingInputMode="File",
                 trainingCompressionCodec=None,
                 trainingMaxRuntimeInSeconds=24*60*60,
                 trainingKmsKeyId=None,
                 modelEnvironmentVariables=None,
                 endpointCreationPolicy=EndpointCreationPolicy.CREATE_ON_CONSTRUCT,
                 sagemakerClient=SageMakerClients.create_sagemaker_client(),
                 region=None,
                 s3Client=SageMakerClients.create_s3_default_client(),
                 stsClient=SageMakerClients.create_sts_default_client(),
                 modelPrependInputRowsToTransformationRows=True,
                 deleteStagingDataAfterTraining=True,
                 namePolicyFactory=RandomNamePolicyFactory(),
                 uid=None):

        if trainingSparkDataFormatOptions is None:
            trainingSparkDataFormatOptions = {}

        if modelEnvironmentVariables is None:
            modelEnvironmentVariables = {}

        if uid is None:
            uid = Identifiable._randomUID()

        kwargs = locals()
        del kwargs['self']
        super(XGBoostSageMakerEstimator, self).__init__(**kwargs)

    def _get_java_obj(self, **kwargs):
        return self._new_java_obj(
            XGBoostSageMakerEstimator._wrapped_class,
            kwargs['sagemakerRole'],
            kwargs['trainingInstanceType'],
            kwargs['trainingInstanceCount'],
            kwargs['endpointInstanceType'],
            kwargs['endpointInitialInstanceCount'],
            kwargs['requestRowSerializer'],
            kwargs['responseRowDeserializer'],
            kwargs['trainingInputS3DataPath'],
            kwargs['trainingOutputS3DataPath'],
            kwargs['trainingInstanceVolumeSizeInGB'],
            Option(kwargs['trainingProjectedColumns']),
            kwargs['trainingChannelName'],
            Option(kwargs['trainingContentType']),
            kwargs['trainingS3DataDistribution'],
            kwargs['trainingSparkDataFormat'],
            kwargs['trainingSparkDataFormatOptions'],
            kwargs['trainingInputMode'],
            Option(kwargs['trainingCompressionCodec']),
            kwargs['trainingMaxRuntimeInSeconds'],
            Option(kwargs['trainingKmsKeyId']),
            kwargs['modelEnvironmentVariables'],
            kwargs['endpointCreationPolicy'],
            kwargs['sagemakerClient'],
            Option(kwargs['region']),
            kwargs['s3Client'],
            kwargs['stsClient'],
            kwargs['modelPrependInputRowsToTransformationRows'],
            kwargs['deleteStagingDataAfterTraining'],
            kwargs['namePolicyFactory'],
            kwargs['uid']
        )

    def getBooster(self):
        return self.getOrDefault(self.booster)

    def setBooster(self, value):
        if value not in ('gbtree', 'gblinear', 'dart'):
            raise ValueError("booster must be 'gbtree', 'gblinear' or 'dart'. got: %s" % value)
        self._set(booster=value)

    def getSilent(self):
        return self.getOrDefault(self.silent)

    def setSilent(self, value):
        if value not in (0, 1):
            raise ValueError("silent must be either 0 or 1. got: %s" % value)
        self._set(silent=value)

    def getNThread(self):
        return self.getOrDefault(self.nthread)

    def setNThread(self, value):
        if value < 1:
            raise ValueError("nthread must be >= 1 got: %s" % value)
        self._set(nthread=value)

    def getEta(self):
        return self.getOrDefault(self.eta)

    def setEta(self, value):
        if value < 0 or value > 1:
            raise ValueError("eta must be within range [0.0, 1.0] got: %s" % value)
        self._set(eta=value)

    def getGamma(self):
        return self.getOrDefault(self.gamma)

    def setGamma(self, value):
        if value < 0:
            raise ValueError("gamma must be >= 0  got: %s" % value)
        self._set(gamma=value)

    def getMaxDepth(self):
        return self.getOrDefault(self.max_depth)

    def setMaxDepth(self, value):
        if value < 0:
            raise ValueError("gamma must be >=0 got: %s" % value)
        self._set(max_depth=value)

    def getMinChildWeight(self):
        return self.getOrDefault(self.min_child_weight)

    def setMinChildWeight(self, value):
        if value < 0:
            raise ValueError("min_child_weight must be >= 0 got: %s" % value)
        self._set(min_child_weight=value)

    def getMaxDeltaStep(self):
        return self.getOrDefault(self.max_delta_step)

    def setMaxDeltaStep(self, value):
        if value < 0:
            raise ValueError("max_delta_weight must be >=0 got: %s" % value)
        self._set(max_delta_step=value)

    def getSubsample(self):
        return self.getOrDefault(self.subsample)

    def setSubsample(self, value):
        if value <= 0 or value > 1:
            raise ValueError("subsample must be in range (0, 1] got: %s" % value)
        self._set(subsample=value)

    def getColSampleByTree(self):
        return self.getOrDefault(self.colsample_bytree)

    def setColSampleByTree(self, value):
        if value <= 0 or value > 1:
            raise ValueError("colsample_bytree must be in range (0, 1] got: %s" % value)
        self._set(colsample_bytree=value)

    def getColSampleByLevel(self):
        return self.getOrDefault(self.colsample_bylevel)

    def setColSampleByLevel(self, value):
        if value <= 0 or value > 1:
            raise ValueError("colsample_by_level must be in range (0, 1] got: %s" % value)
        self._set(colsample_bylevel=value)

    def getLambda(self):
        return self.getOrDefault(self._lambda)

    def setLambda(self, value):
        self._set(_lambda=value)

    def getAlpha(self):
        return self.getOrDefault(self.alpha)

    def setAlpha(self, value):
        self._set(alpha=value)

    def getTreeMethod(self):
        return self.getOrDefault(self.tree_method)

    def setTreeMethod(self, value):
        if value not in ("auto", "exact", "approx", "hist"):
            raise ValueError("tree_method must be one of: 'auto', 'exact', 'approx', 'hist', "
                             "got: %s" % value)
        self._set(tree_method=value)

    def getSketchEps(self):
        return self.getOrDefault(self.sketch_eps)

    def setSketchEps(self, value):
        if value <= 0 or value >= 1:
            raise ValueError("sketch_eps must be in range (0, 1) got: %s" % value)
        self._set(sketch_eps=value)

    def getScalePosWeight(self):
        return self.getOrDefault(self.scale_pos_weight)

    def setScalePosWeight(self, value):
        self._set(scale_pos_weight=value)

    def getUpdater(self):
        return self.getOrDefault(self.updater)

    def setUpdater(self, value):
        valid_tokens = ("grow_colmaker", "distcol", "grow_histmaker", "grow_local_histmaker",
                        "grow_skmaker", "sync", "refresh", "prune")
        tokens = value.split(",")
        for token in tokens:
            if token.strip() not in valid_tokens:
                raise ValueError("values allowed in updater are: %s, found: %s " %
                                 (','.join(valid_tokens), token))
        self._set(updater=value)

    def getRefreshLeaf(self):
        return self.getOrDefault(self.refresh_leaf)

    def setRefreshLeaf(self, value):
        if value not in (0, 1):
            raise ValueError("refresh_leaf must be either 0 or 1, got: %s" % value)
        self._set(refresh_leaf=value)

    def getProcessType(self):
        return self.getOrDefault(self.process_type)

    def setProcessType(self, value):
        if value not in ("default", "update"):
            raise ValueError("process_type must be 'default' or 'update', got: %s" % value)
        self._set(process_type=value)

    def getGrowPolicy(self):
        return self.getOrDefault(self.grow_policy)

    def setGrowPolicy(self, value):
        if value not in ("depthwise", "lossguide"):
            raise ValueError("grow_policy must be 'depthwise' or 'lossguide', got: %s" % value)
        self._set(grow_policy=value)

    def getMaxLeaves(self):
        return self.getOrDefault(self.max_leaves)

    def setMaxLeaves(self, value):
        if value < 0:
            raise ValueError("max_leaves must be >=0, got: %s" % value)
        self._set(max_leaves=value)

    def getMaxBin(self):
        return self.getOrDefault(self.max_bin)

    def setMaxBin(self, value):
        if value < 1:
            raise ValueError("max_bin must be >=1, got: %s" % value)
        self._set(max_bin=value)

    def getSampleType(self):
        return self.getOrDefault(self.sample_type)

    def setSampleType(self, value):
        if value not in ("uniform", "weighted"):
            raise ValueError("sample_type must be 'uniform' or 'weighted', got: %s" % value)
        self._set(sample_type=value)

    def getNormalizeType(self):
        return self.getOrDefault(self.normalize_type)

    def setNormalizeType(self, value):
        if value not in ("tree", "forest"):
            raise ValueError("normalize_type must be 'tree' or 'forest', got: %s" % value)
        self._set(normalize_type=value)

    def getRateDrop(self):
        return self.getOrDefault(self.rate_drop)

    def setRateDrop(self, value):
        if value < 0 or value > 1:
            raise ValueError("rate_drop must be in range [0.0, 1.0], got: %s" % value)
        self._set(rate_drop=value)

    def getOneDrop(self):
        return self.getOrDefault(self.one_drop)

    def setOneDrop(self, value):
        if value not in (0, 1):
            raise ValueError("one_drop must be 0 or 1, got: %s" % value)
        self._set(one_drop=value)

    def getSkipDrop(self):
        return self.getOrDefault(self.skip_drop)

    def setSkipDrop(self, value):
        if value < 0 or value > 1:
            raise ValueError("skip_drop must be in range [0.0, 1.0], got: %s" % value)
        self._set(skip_drop=value)

    def getLambdaBias(self):
        return self.getOrDefault(self.lambda_bias)

    def setLambdaBias(self, value):
        if value < 0 or value > 1:
            raise ValueError("lambda_bias must in range [0.0, 1.0], got: %s" % value)
        self._set(lambda_bias=value)

    def getTweedieVariancePower(self):
        return self.getOrDefault(self.tweedie_variance_power)

    def setTweedieVariancePower(self, value):
        if value <= 1 or value >= 2:
            raise ValueError("tweedie_variance_power must be in range (1.0, 2.0), got: %s" % value)
        self._set(tweedie_variance_power=value)

    def getObjective(self):
        return self.getOrDefault(self.objective)

    def setObjective(self, value):
        allowed_values = ("reg:linear", "reg:logistic", "binary:logistic", "binary:logistraw",
                          "count:poisson", "multi:softmax", "multi:softprob", "rank:pairwise",
                          "reg:gamma", "reg:tweedie")

        if value not in allowed_values:
            raise ValueError("objective must be one of (%s), got: %s" %
                             (','.join(allowed_values), value))
        self._set(objective=value)

    def getNumClasses(self):
        return self.getOrDefault(self.num_class)

    def setNumClasses(self, value):
        if value < 1:
            raise ValueError("num_class must be >=1, got: %s" % value)
        self._set(num_class=value)

    def getBaseScore(self):
        return self.getOrDefault(self.base_score)

    def setBaseScore(self, value):
        self._set(base_score=value)

    def getEvalMetric(self):
        return self.getOrDefault(self.eval_metric)

    def setEvalMetric(self, value):
        allowed_values = ("rmse", "mae", "logloss", "error", "error@t", "merror",
                          "mlogloss", "auc", "ndcg", "map", "ndcg@n", "ndcg-", "ndcg@n-",
                          "map-", "map@n-")

        if value not in allowed_values:
            raise ValueError("eval_metric must be one of (%s), got: %s" %
                             (','.join(allowed_values), value))
        self._set(eval_metric=value)

    def getSeed(self):
        return self.getOrDefault(self.seed)

    def setSeed(self, value):
        self._set(seed=value)

    def getNumRound(self):
        return self.getOrDefault(self.num_round)

    def setNumRound(self, value):
        if value < 1:
            raise ValueError("num_round must be  >= 1, got: %s" % value)
        self._set(num_round=value)

    @classmethod
    def _from_java(cls, javaObject):
        return XGBoostSageMakerEstimator(sagemakerRole=None, javaObject=javaObject)
