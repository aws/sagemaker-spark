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
from sagemaker_pyspark.transformation.deserializers import (
    FactorizationMachinesBinaryClassifierDeserializer,
    FactorizationMachinesRegressorDeserializer)


class FactorizationMachinesParams(Params):

    feature_dim = Param(Params._dummy(), "feature_dim",
                        "The dimension of the input vectors. Must be > 0. ",
                        typeConverter=TypeConverters.toInt)

    num_factors = Param(Params._dummy(), "num_factors",
                        "Dimensionality of factorization. Must be > 0. ",
                        typeConverter=TypeConverters.toInt)

    predictor_type = Param(Params._dummy(), "predictor_type",
                           "Whether training is for binary classification or regression. "
                           "Supported options: 'binary_classifier', and 'regressor'. ",
                           typeConverter=TypeConverters.toString)

    mini_batch_size = Param(Params._dummy(), "mini_batch_size",
                            "The number of examples in a mini-batch. Must be > 0. ",
                            typeConverter=TypeConverters.toInt)

    epochs = Param(Params._dummy(), "epochs",
                   "The number of passes done over the training data. Must be > 0. ",
                   typeConverter=TypeConverters.toInt)

    clip_gradient = Param(Params._dummy(), "clip_gradient",
                          "Clip the gradient by projecting onto"
                          "the box [-clip_gradient, +clip_gradient]. ",
                          typeConverter=TypeConverters.toFloat)

    eps = Param(Params._dummy(), "eps",
                "Small value to avoid division by 0. ",
                typeConverter=TypeConverters.toFloat)

    rescale_grad = Param(Params._dummy(), "rescale_grad",
                         "Multiplies the gradient with this value before updating. ",
                         typeConverter=TypeConverters.toFloat)

    bias_lr = Param(Params._dummy(), "bias_lr",
                    "Multiplies the gradient with this value before updating. Must be >= 0. ",
                    typeConverter=TypeConverters.toFloat)

    linear_lr = Param(Params._dummy(), "linear_lr",
                      "Learning rate for linear terms. Must be >= 0. ",
                      typeConverter=TypeConverters.toFloat)

    factors_lr = Param(Params._dummy(), "factors_lr",
                       "Learning rate for factorization terms. Must be >= 0. ",
                       typeConverter=TypeConverters.toFloat)

    bias_wd = Param(Params._dummy(), "bias_wd",
                    "Weight decay for the bias term. Must be >= 0. ",
                    typeConverter=TypeConverters.toFloat)

    linear_wd = Param(Params._dummy(), "linear_wd",
                      "Weight decay for linear terms. Must be >= 0. ",
                      typeConverter=TypeConverters.toFloat)

    factors_wd = Param(Params._dummy(), "factors_wd",
                       "Weight decay for factorization terms. Must be >= 0. ",
                       typeConverter=TypeConverters.toFloat)

    bias_init_method = Param(Params._dummy(), "bias_init_method",
                             "Initialization method for the bias supports"
                             " 'normal', 'uniform' and 'constant'. ",
                             typeConverter=TypeConverters.toString)

    bias_init_scale = Param(Params._dummy(), "bias_init_scale",
                            "Range for bias term uniform initialization. Must be >= 0. ",
                            typeConverter=TypeConverters.toFloat)

    bias_init_sigma = Param(Params._dummy(), "bias_init_sigma",
                            "Standard deviation to initialize bias terms. Must be >= 0. ",
                            typeConverter=TypeConverters.toFloat)

    bias_init_value = Param(Params._dummy(), "bias_init_value",
                            "Initial value for the bias term. ",
                            typeConverter=TypeConverters.toFloat)

    linear_init_method = Param(Params._dummy(), "linear_init_method",
                               "Initialization method for linear term,"
                               " supports: 'normal', 'uniform' and 'constant'. ",
                               typeConverter=TypeConverters.toString)

    linear_init_scale = Param(Params._dummy(), "linear_init_scale",
                              "Range for linear term uniform initialization. Must be >= 0. ",
                              typeConverter=TypeConverters.toFloat)

    linear_init_sigma = Param(Params._dummy(), "linear_init_sigma",
                              "Standard deviation to initialize linear terms. Must be >= 0. ",
                              typeConverter=TypeConverters.toFloat)

    linear_init_value = Param(Params._dummy(), "linear_init_value",
                              "Initial value for linear term. ",
                              typeConverter=TypeConverters.toFloat)

    factors_init_method = Param(Params._dummy(), "factors_init_method",
                                "Init method for factorization terms,"
                                " supports: 'normal', 'uniform' and 'constant'. ",
                                typeConverter=TypeConverters.toString)

    factors_init_scale = Param(Params._dummy(), "factors_init_scale",
                               "Range for factorization terms uniform initialization."
                               " Must be >= 0. ",
                               typeConverter=TypeConverters.toFloat)

    factors_init_sigma = Param(Params._dummy(), "factors_init_sigma",
                               "Standard deviation to initialize factorization terms."
                               " Must be >= 0. ",
                               typeConverter=TypeConverters.toFloat)

    factors_init_value = Param(Params._dummy(), "factors_init_value",
                               "Initial value for factorization term. ",
                               typeConverter=TypeConverters.toFloat)

    def getFeatureDim(self):
        return self.getOrDefault(self.feature_dim)

    def setFeatureDim(self, value):
        if value <= 0:
            raise ValueError("feature_dim must be > 0. Got %s" % value)
        self._set(feature_dim=value)

    def getNumFactors(self):
        return self.getOrDefault(self.num_factors)

    def setNumFactors(self, value):
        if value <= 0:
            raise ValueError("num_factors must be > 0, got: %s" % value)
        self._set(num_factors=value)

    def getMiniBatchSize(self):
        return self.getOrDefault(self.mini_batch_size)

    def setMiniBatchSize(self, value):
        if value <= 0:
            raise ValueError("mini_batch_size must be > 0. Got %s" % value)
        self._set(mini_batch_size=value)

    def getEpochs(self):
        return self.getOrDefault(self.epochs)

    def setEpochs(self, value):
        if value <= 0:
            raise ValueError("epochs must be > 0, got: %s" % value)
        self._set(epochs=value)

    def getClipGradient(self):
        return self.getOrDefault(self.clip_gradient)

    def setClipGradient(self, value):
        self._set(clip_gradient=value)

    def getEps(self):
        return self.getOrDefault(self.eps)

    def setEps(self, value):
        self._set(eps=value)

    def getRescaleGrad(self):
        return self.getOrDefault(self.rescale_grad)

    def setRescaleGrad(self, value):
        self._set(rescale_grad=value)

    def getBiasLr(self):
        return self.getOrDefault(self.bias_lr)

    def setBiasLr(self, value):
        if value < 0:
            raise ValueError("bias_lr must be >= 0. Got %s" % value)
        self._set(bias_lr=value)

    def getLinearLr(self):
        return self.getOrDefault(self.linear_lr)

    def setLinearLr(self, value):
        if value < 0:
            raise ValueError("linear_lr must be >= 0. Got %s" % value)
        self._set(linear_lr=value)

    def getFactorsLr(self):
        return self.getOrDefault(self.factors_lr)

    def setFactorsLr(self, value):
        if value < 0:
            raise ValueError("factors_lr must be >= 0. Got %s" % value)
        self._set(factors_lr=value)

    def getBiasWd(self):
        return self.getOrDefault(self.bias_wd)

    def setBiasWd(self, value):
        if value < 0:
            raise ValueError("bias_wd must be >= 0. Got %s" % value)
        self._set(bias_wd=value)

    def getLinearWd(self):
        return self.getOrDefault(self.linear_wd)

    def setLinearWd(self, value):
        if value < 0:
            raise ValueError("linear_wd must be >= 0. Got %s" % value)
        self._set(linear_wd=value)

    def getFactorsWd(self):
        return self.getOrDefault(self.factors_wd)

    def setFactorsWd(self, value):
        if value < 0:
            raise ValueError("factors_wd must be >= 0. Got %s" % value)
        self._set(factors_wd=value)

    def getBiasInitMethod(self):
        return self.getOrDefault(self.bias_init_method)

    def setBiasInitMethod(self, value):
        if value not in ('uniform', 'normal', 'constant'):
            raise ValueError("bias_init_method must be 'uniform',"
                             " 'constant' or 'normal', got: %s" % value)
        self._set(bias_init_method=value)

    def getBiasInitScale(self):
        return self.getOrDefault(self.bias_init_scale)

    def setBiasInitScale(self, value):
        if value < 0:
            raise ValueError("bias_init_scale must be >= 0. Got %s" % value)
        self._set(bias_init_scale=value)

    def getBiasInitSigma(self):
        return self.getOrDefault(self.bias_init_sigma)

    def setBiasInitSigma(self, value):
        if value < 0:
            raise ValueError("bias_init_sigma must be >= 0. Got %s" % value)
        self._set(bias_init_sigma=value)

    def getBiasInitValue(self):
        return self.getOrDefault(self.bias_init_value)

    def setBiasInitValue(self, value):
        self._set(bias_init_value=value)

    def getLinearInitMethod(self):
        return self.getOrDefault(self.linear_init_method)

    def setLinearInitMethod(self, value):
        if value not in ('uniform', 'normal', 'constant'):
            raise ValueError("linear_init_method must be 'uniform', "
                             "'constant' or 'normal', got: %s" % value)
        self._set(linear_init_method=value)

    def getLinearInitScale(self):
        return self.getOrDefault(self.linear_init_scale)

    def setLinearInitScale(self, value):
        if value < 0:
            raise ValueError("linear_init_scale must be >= 0. Got %s" % value)
        self._set(linear_init_scale=value)

    def getLinearInitSigma(self):
        return self.getOrDefault(self.linear_init_sigma)

    def setLinearInitSigma(self, value):
        if value < 0:
            raise ValueError("linear_init_sigma must be >= 0. Got %s" % value)
        self._set(linear_init_sigma=value)

    def getLinearInitValue(self):
        return self.getOrDefault(self.linear_init_value)

    def setLinearInitValue(self, value):
        self._set(linear_init_value=value)

    def getFactorsInitMethod(self):
        return self.getOrDefault(self.factors_init_method)

    def setFactorsInitMethod(self, value):
        if value not in ('uniform', 'normal', 'constant'):
            raise ValueError("factors_init_method must be 'uniform', "
                             "'constant' or 'normal', got: %s" % value)
        self._set(factors_init_method=value)

    def getFactorsInitScale(self):
        return self.getOrDefault(self.factors_init_scale)

    def setFactorsInitScale(self, value):
        if value < 0:
            raise ValueError("factors_init_scale must be >= 0. Got %s" % value)
        self._set(factors_init_scale=value)

    def getFactorsInitSigma(self):
        return self.getOrDefault(self.factors_init_sigma)

    def setFactorsInitSigma(self, value):
        if value < 0:
            raise ValueError("factors_init_sigma must be >= 0. Got %s" % value)
        self._set(factors_init_sigma=value)

    def getFactorsInitValue(self):
        return self.getOrDefault(self.factors_init_value)

    def setFactorsInitValue(self, value):
        self._set(factors_init_value=value)


class FactorizationMachinesBinaryClassifier(SageMakerEstimatorBase, FactorizationMachinesParams):
    """
    A :class:`~sagemaker_pyspark.SageMakerEstimator` that runs a Factorization Machines training
    job in "binary classifier" mode in SageMaker and returns a
    :class:`~sagemaker_pyspark.SageMakerModel` that can be used to transform a DataFrame using
    the hosted Factorization Machines model. The Factorization Machines Binary Classifier is useful
    for classifying examples into one of two classes.

    Amazon SageMaker Factorization Machines trains on RecordIO-encoded Amazon Record protobuf data.
    SageMaker pyspark writes a DataFrame to S3 by selecting a column of Vectors named "features"
    and, if present, a column of Doubles named "label". These names are configurable by passing a
    dictionary with entries in trainingSparkDataFormatOptions with key "labelColumnName" or
    "featuresColumnName", with values corresponding to the desired label and features columns.

    Inferences made against an Endpoint hosting a Factorization Machines Binary classifier model
    contain a "score" field and a "predicted_label" field, both appended to the
    input DataFrame as Doubles.

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
        "com.amazonaws.services.sagemaker.sparksdk.algorithms."\
        "FactorizationMachinesBinaryClassifier"

    def __init__(
            self,
            trainingInstanceType,
            trainingInstanceCount,
            endpointInstanceType,
            endpointInitialInstanceCount,
            sagemakerRole=IAMRoleFromConfig(),
            requestRowSerializer=ProtobufRequestRowSerializer(),
            responseRowDeserializer=FactorizationMachinesBinaryClassifierDeserializer(),
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
            uid=None,
            javaObject=None):

        if trainingSparkDataFormatOptions is None:
            trainingSparkDataFormatOptions = {}

        if modelEnvironmentVariables is None:
            modelEnvironmentVariables = {}

        if uid is None:
            uid = Identifiable._randomUID()

        kwargs = locals().copy()
        del kwargs['self']

        super(FactorizationMachinesBinaryClassifier, self).__init__(**kwargs)

        default_params = {
            'predictor_type': 'binary_classifier'
        }

        self._setDefault(**default_params)

    def _get_java_obj(self, **kwargs):
        if 'javaObject' in kwargs and kwargs['javaObject'] is not None:
            return kwargs['javaObject']
        else:
            return self._new_java_obj(
                FactorizationMachinesBinaryClassifier._wrapped_class,
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

    @classmethod
    def _from_java(cls, javaObject):
        return FactorizationMachinesBinaryClassifier(sagemakerRole=None, javaObject=javaObject)


class FactorizationMachinesRegressor(SageMakerEstimatorBase, FactorizationMachinesParams):
    """
    A :class:`~sagemaker_pyspark.SageMakerEstimator` that runs a Factorization Machines training
    job in "regressor" mode in SageMaker and returns a  :class:`~sagemaker_pyspark.SageMakerModel`
    that can be used to transform a DataFrame using the hosted Linear Learner model.
    The Factorization Machines Regressor is useful for predicting a real-valued label
    from training examples.

    Amazon SageMaker Linear Learner trains on RecordIO-encoded Amazon Record protobuf data.
    SageMaker pyspark writes a DataFrame to S3 by selecting a column of Vectors named "features"
    and, if present, a column of Doubles named "label". These names are configurable by passing a
    dictionary with entries in trainingSparkDataFormatOptions with key "labelColumnName" or
    "featuresColumnName", with values corresponding to the desired label and features columns.

    For inference against a hosted Endpoint, the SageMakerModel returned by :meth :`fit()` by
    Factorization Machines uses :class:`~sagemaker_pyspark.transformation
    .serializers.ProtobufRequestRowSerializer` to serialize Rows into RecordIO-encoded Amazon
    Record protobuf messages, by default selecting the column named "features" expected to contain
    a Vector of Doubles.

    Inferences made against an Endpoint hosting a Factorization Machines Regressor model contain
    a "score" field appended to the input DataFrame as a Double.

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
    _wrapped_class = "com.amazonaws.services.sagemaker.sparksdk.algorithms."\
                     "FactorizationMachinesRegressor"

    def __init__(self,
                 trainingInstanceType,
                 trainingInstanceCount,
                 endpointInstanceType,
                 endpointInitialInstanceCount,
                 sagemakerRole=IAMRoleFromConfig(),
                 requestRowSerializer=ProtobufRequestRowSerializer(),
                 responseRowDeserializer=FactorizationMachinesRegressorDeserializer(),
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
                 uid=None,
                 javaObject=None):

        if trainingSparkDataFormatOptions is None:
            trainingSparkDataFormatOptions = {}

        if modelEnvironmentVariables is None:
            modelEnvironmentVariables = {}

        if uid is None:
            uid = Identifiable._randomUID()

        kwargs = locals().copy()
        del kwargs['self']
        super(FactorizationMachinesRegressor, self).__init__(**kwargs)

        default_params = {
            'predictor_type': 'regressor'
        }

        self._setDefault(**default_params)

    def _get_java_obj(self, **kwargs):
        if 'javaObject' in kwargs and kwargs['javaObject'] is not None:
            return kwargs['javaObject']
        else:
            return self._new_java_obj(
                FactorizationMachinesRegressor._wrapped_class,
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

    @classmethod
    def _from_java(cls, javaObject):
        return FactorizationMachinesRegressor(sagemakerRole=None, javaObject=javaObject)
