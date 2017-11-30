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

import numbers
from pyspark.ml.param import Params, Param, TypeConverters
from pyspark.ml.util import Identifiable

from sagemaker_pyspark import (SageMakerEstimatorBase, S3AutoCreatePath, Option, IAMRoleFromConfig,
                               EndpointCreationPolicy, SageMakerClients, RandomNamePolicyFactory)

from sagemaker_pyspark.transformation.serializers import ProtobufRequestRowSerializer
from sagemaker_pyspark.transformation.deserializers import KMeansProtobufResponseRowDeserializer


class KMeansSageMakerEstimator(SageMakerEstimatorBase):
    """
    A :class:`~sagemaker_pyspark.SageMakerEstimator` that runs a KMeans training job on
    Amazon SageMaker upon a call to :meth:`.fit` and returns a
    :class:`~sagemaker_pyspark.SageMakerModel` that can be used to transform a DataFrame using
    the hosted K-Means model. K-Means Clustering is useful for grouping similar examples in your
    dataset.

    Amazon SageMaker K-Means clustering trains on RecordIO-encoded Amazon Record protobuf data.
    SageMaker pyspark writes a DataFrame to S3 by selecting a column of Vectors named "features"
    and, if present, a column of Doubles named "label". These names are configurable by passing a
    dictionary with entries in `trainingSparkDataFormatOptions` with key "labelColumnName" or
    "featuresColumnName", with values corresponding to the desired label and features columns.

    For inference, the SageMakerModel returned by :meth:`fit()` by the KMeansSageMakerEstimator uses
    :class:`~sagemaker_pyspark.transformation.serializers.ProtobufRequestRowSerializer` to
    serialize Rows into RecordIO-encoded Amazon Record protobuf messages for inference, by default
    selecting the column named "features" expected to contain a Vector of Doubles.

    Inferences made against an Endpoint hosting a K-Means model contain a "closest_cluster" field
    and a "distance_to_cluster" field, both appended to the input DataFrame as columns of Double.

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
    _wrapped_class = "com.amazonaws.services.sagemaker.sparksdk.algorithms.KMeansSageMakerEstimator"

    k = Param(Params._dummy(),
              "k",
              "The number of clusters to create. Must be > 1",
              typeConverter=TypeConverters.toInt)

    init_method = Param(Params._dummy(),
                        "init_method",
                        "The initialization algorithm to choose centroids. "
                        "Supported options: 'random' and 'kmeans++'.",
                        typeConverter=TypeConverters.toString)

    local_lloyd_max_iter = Param(Params._dummy(),
                                 "local_lloyd_max_iter",
                                 "Maximum iterations for LLoyds EM procedure "
                                 "in the local kmeans used in finalized stage. Must be > 0",
                                 typeConverter=TypeConverters.toInt)

    local_lloyd_tol = Param(Params._dummy(),
                            "local_lloyd_tol",
                            "Tolerance for change in ssd for early stopping "
                            "in the local kmeans. Must be in range [0, 1].",
                            typeConverter=TypeConverters.toFloat)

    local_lloyd_num_trials = Param(Params._dummy(),
                                   "local_lloyd_num_trials",
                                   "The number of trials of the local kmeans "
                                   "algorithm. Must be > 0 or 'auto'",
                                   typeConverter=TypeConverters.toString)

    local_lloyd_init_method = Param(Params._dummy(),
                                    "local_lloyd_init_method",
                                    "The local initialization algorithm to choose centroids. "
                                    "Supported options: 'random' and 'kmeans++'",
                                    typeConverter=TypeConverters.toString)

    half_life_time_size = Param(Params._dummy(), "half_life_time_size",
                                "The weight decaying rate of each point. Must be >= 0",
                                typeConverter=TypeConverters.toInt)

    epochs = Param(Params._dummy(), "epochs",
                   "The number of passes done over the training data. Must be > 0",
                   typeConverter=TypeConverters.toInt)

    extra_center_factor = Param(Params._dummy(), "extra_center_factor",
                                "The factor of extra centroids to create. "
                                "Must be > 0 or 'auto'",
                                typeConverter=TypeConverters.toString)

    mini_batch_size = Param(Params._dummy(), "mini_batch_size",
                            "The number of examples in a mini-batch. Must be > 0",
                            typeConverter=TypeConverters.toInt)

    feature_dim = Param(Params._dummy(), "feature_dim",
                        "The dimension of the input vectors. Must be > 0",
                        typeConverter=TypeConverters.toInt)

    eval_metrics = Param(Params._dummy(), "eval_metrics",
                         "Metric to be used for scoring the model. String of comma separated"
                         " metrics. Supported metrics are 'msd' and 'ssd'."
                         " 'msd' Means Square Error, 'ssd': Sum of square distance",
                         typeConverter=TypeConverters.toString)

    def __init__(self,
                 trainingInstanceType,
                 trainingInstanceCount,
                 endpointInstanceType,
                 endpointInitialInstanceCount,
                 sagemakerRole=IAMRoleFromConfig(),
                 requestRowSerializer=ProtobufRequestRowSerializer(),
                 responseRowDeserializer=KMeansProtobufResponseRowDeserializer(),
                 trainingInputS3DataPath=S3AutoCreatePath(),
                 trainingOutputS3DataPath=S3AutoCreatePath(),
                 trainingInstanceVolumeSizeInGB=1024,
                 trainingProjectedColumns=None,
                 trainingChannelName="train",
                 trainingContentType=None,
                 trainingS3DataDistribution="ShardedByS3Key",
                 trainingSparkDataFormat="sagemaker",
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

        kwargs = locals().copy()
        del kwargs['self']

        super(KMeansSageMakerEstimator, self).__init__(**kwargs)

        default_params = {
            'k': 2
        }

        self._setDefault(**default_params)

    def _get_java_obj(self, **kwargs):
        return self._new_java_obj(
            KMeansSageMakerEstimator._wrapped_class,
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

    def getK(self):
        return self.getOrDefault(self.k)

    def setK(self, value):
        if value < 2:
            raise ValueError("K must be >= 2, got: %s" % value)
        self._set(k=value)

    def getMaxIter(self):
        return self.getOrDefault(self.local_lloyd_max_iter)

    def setMaxIter(self, value):
        if value < 1:
            raise ValueError("MaxIter must be > 0, got: %s" % value)
        self._set(local_lloyd_max_iter=value)

    def getTol(self):
        return self.getOrDefault(self.local_lloyd_tol)

    def setTol(self, value):
        if value > 1 or value < 0:
            raise ValueError("Tol must be within [0, 1], got: %s" % value)
        self._set(local_lloyd_tol=value)

    def getLocalInitMethod(self):
        return self.getOrDefault(self.local_lloyd_init_method)

    def setLocalInitMethod(self, value):
        if value not in ('random', 'kmeans++'):
            raise ValueError("LocalInitMethod must be 'random' or 'kmeans++', got %s" % value)
        self._set(local_lloyd_init_method=value)

    def getHalflifeTime(self):
        return self.getOrDefault(self.half_life_time_size)

    def setHalflifeTime(self, value):
        if value < 0:
            raise ValueError("HalflifeTime must be >=0, got: %s" % value)
        self._set(half_life_time_size=value)

    def getEpochs(self):
        return self.getOrDefault(self.epochs)

    def setEpochs(self, value):
        if value < 1:
            raise ValueError("Epochs must be > 0, got: %s" % value)
        self._set(epochs=value)

    def getInitMethod(self):
        return self.getOrDefault(self.init_method)

    def setInitMethod(self, value):
        if value not in ('random', 'kmeans++'):
            raise ValueError("InitMethod must be 'random' or 'kmeans++', got: %s" % value)
        self._set(init_method=value)

    def getCenterFactor(self):
        return self.getOrDefault(self.extra_center_factor)

    def setCenterFactor(self, value):
        if isinstance(value, numbers.Real) and value < 1:
            raise ValueError("CenterFactor must be 'auto' or > 0, got: %s" % value)
        if value != 'auto' and int(value) < 1:
            raise ValueError("CenterFactor must be 'auto' or > 0, got: %s" % value)
        self._set(extra_center_factor=str(value))

    def getTrialNum(self):
        return self.getOrDefault(self.local_lloyd_num_trials)

    def setTrialNum(self, value):
        if isinstance(value, numbers.Real) and value < 1:
            raise ValueError("TrialNum must be 'auto' or > 0, got: %s" % value)
        if value != 'auto' and int(value) < 1:
            raise ValueError("TrialNum must be 'auto' or > 0, got: %s" % value)
        self._set(local_lloyd_num_trials=str(value))

    def getMiniBatchSize(self):
        return self.getOrDefault(self.mini_batch_size)

    def setMiniBatchSize(self, size):
        if size <= 0:
            raise ValueError("mini_batch_size must be > 0. Got %s" % size)
        self._set(mini_batch_size=size)

    def getFeatureDim(self):
        return self.getOrDefault(self.feature_dim)

    def setFeatureDim(self, value):
        if value <= 0:
            raise ValueError("feature_dim must be > 0. Got %s" % value)
        self._set(feature_dim=value)

    def getEvalMetrics(self):
        return self.getOrDefault(self.eval_metrics).strip('[').strip(']')

    def setEvalMetrics(self, value):
        valid_tokens = ("msd", "ssd")
        tokens = value.split(",")
        for token in tokens:
            if token.strip() not in valid_tokens:
                raise ValueError("values allowed in eval_metrics are: %s, found: %s " %
                                 (','.join(valid_tokens), token))
        self._set(eval_metrics='[' + value + ']')

    @classmethod
    def _from_java(cls, javaObject):
        return KMeansSageMakerEstimator(sagemakerRole=None, javaObject=javaObject)
