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

from sagemaker_pyspark.transformation.serializers import ProtobufRequestRowSerializer
from sagemaker_pyspark.transformation.deserializers import LDAProtobufResponseRowDeserializer


class LDASageMakerEstimator(SageMakerEstimatorBase):
    """
    A :class:`~sagemaker_pyspark.SageMakerEstimator` runs a LDA training job on Amazon SageMaker
    upon a call to :meth:`.fit` on a DataFrame and returns a
    :class:`~sagemaker_pyspark.SageMakerModel`. LDA is unsupervised learning algorithm that
    attempts to describe a set of observations as a mixture of
    distinct categories. LDA is most commonly used to discover a user-specified number of topics
    shared by documents within a text corpus.

    Amazon SageMaker LDA trains on RecordIO-encoded Amazon Record protobuf data.
    SageMaker pyspark writes a DataFrame to S3 by selecting a column of Vectors named "features".
    These names are configurable by passing
    a map with entries in trainingSparkDataFormatOptions with key "featuresColumnName",
    with values corresponding to the desired feature column.

    For inference, the SageMakerModel returned by :meth:`fit()` by the LDASageMakerEstimator uses
    :class:`~sagemaker_pyspark.transformation.serializers.ProtobufRequestRowSerializer` to
    serialize Rows into RecordIO-encoded Amazon Record protobuf messages for inference, by default
    selecting the column named "features" expected to contain a Vector of Doubles.

    Inferences made against an Endpoint hosting a LDA model contain a "topic_mixture" field
    that holds a vector of Double values.

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
    _wrapped_class = "com.amazonaws.services.sagemaker.sparksdk.algorithms.LDASageMakerEstimator"

    num_topics = Param(Params._dummy(),
                       "num_topics",
                       "The number of topics for LDA to find within the data. Must be > 0.",
                       typeConverter=TypeConverters.toInt)

    alpha0 = Param(Params._dummy(),
                   "alpha0",
                   "Initial guess for the concentration parameter: the sum of the "
                   "elements of the Dirichlet prior. "
                   "Must be > 0.",
                   typeConverter=TypeConverters.toFloat)

    max_restarts = Param(Params._dummy(),
                         "max_restarts",
                         "The number of restarts during decomposition phase. Must be > 0.",
                         typeConverter=TypeConverters.toInt)

    max_iterations = Param(Params._dummy(),
                           "max_iterations",
                           "The maximum number of iterations to perform during the ALS phase "
                           "of the algorithm. Must be > 0.",
                           typeConverter=TypeConverters.toInt)

    tol = Param(Params._dummy(),
                "tol",
                "Target error tolerance for the ALS phase of the algorithm. Must be > 0.",
                typeConverter=TypeConverters.toFloat)

    mini_batch_size = Param(Params._dummy(), "mini_batch_size",
                            "The total number of documents in the input document corpus. "
                            "Must be > 0",
                            typeConverter=TypeConverters.toInt)

    feature_dim = Param(Params._dummy(), "feature_dim",
                        "The size of the vocabulary of the input document corpus. Must be > 0",
                        typeConverter=TypeConverters.toInt)

    def __init__(self,
                 trainingInstanceType,
                 trainingInstanceCount,
                 endpointInstanceType,
                 endpointInitialInstanceCount,
                 sagemakerRole=IAMRoleFromConfig(),
                 requestRowSerializer=ProtobufRequestRowSerializer(),
                 responseRowDeserializer=LDAProtobufResponseRowDeserializer(),
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

        super(LDASageMakerEstimator, self).__init__(**kwargs)

    def _get_java_obj(self, **kwargs):
        return self._new_java_obj(
            LDASageMakerEstimator._wrapped_class,
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

    def getNumTopics(self):
        return self.getOrDefault(self.num_topics)

    def setNumTopics(self, value):
        if value < 1:
            raise ValueError("num_topics must be > 0, got: %s" % value)
        self._set(num_topics=value)

    def getAlpha0(self):
        return self.getOrDefault(self.alpha0)

    def setAlpha0(self, value):
        if value <= 0:
            raise ValueError("alpha0 must be > 0, got: %s" % value)
        self._set(alpha0=value)

    def getMaxRestarts(self):
        return self.getOrDefault(self.max_restarts)

    def setMaxRestarts(self, value):
        if value < 1:
            raise ValueError("max_restarts must be > 0, got: %s" % value)
        self._set(max_restarts=value)

    def getMaxIterations(self):
        return self.getOrDefault(self.max_iterations)

    def setMaxIterations(self, value):
        if value < 1:
            raise ValueError("max_iterations must be > 0, got: %s" % value)
        self._set(max_iterations=value)

    def getTol(self):
        return self.getOrDefault(self.tol)

    def setTol(self, value):
        if value <= 0:
            raise ValueError("tol must be > 0, got: %s" % value)
        self._set(tol=value)

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

    @classmethod
    def _from_java(cls, javaObject):
        return LDASageMakerEstimator(sagemakerRole=None, javaObject=javaObject)
