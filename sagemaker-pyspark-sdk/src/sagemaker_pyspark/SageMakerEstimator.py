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

from abc import ABCMeta, abstractmethod

from pyspark import keyword_only
from pyspark.ml.util import Identifiable
from pyspark.ml.wrapper import JavaEstimator

from sagemaker_pyspark import SageMakerJavaWrapper, RandomNamePolicyFactory, SageMakerClients, \
    IAMRoleFromConfig, S3AutoCreatePath, Option

_sagemaker_spark_sdk_package = "com.amazonaws.services.sagemaker.sparksdk"


class EndpointCreationPolicy(object):
    """Determines whether and when to create the Endpoint and other Hosting resources.

    Attributes:
        CREATE_ON_CONSTRUCT: create the Endpoint upon creation of the SageMakerModel, at the end
            of fit()
        CREATE_ON_TRANSFORM: create the Endpoint upon invocation of SageMakerModel.transform().
        DO_NOT_CREATE: do not create the Endpoint.

    """
    class _CreateOnConstruct(SageMakerJavaWrapper):
        _wrapped_class = \
            "%s.EndpointCreationPolicy.CREATE_ON_CONSTRUCT" % _sagemaker_spark_sdk_package

    class _CreateOnTransform(SageMakerJavaWrapper):
        _wrapped_class = \
            "%s.EndpointCreationPolicy.CREATE_ON_TRANSFORM" % _sagemaker_spark_sdk_package

    class _DoNotCreate(SageMakerJavaWrapper):
        _wrapped_class = "%s.EndpointCreationPolicy.DO_NOT_CREATE" % _sagemaker_spark_sdk_package

    CREATE_ON_CONSTRUCT = _CreateOnConstruct()
    CREATE_ON_TRANSFORM = _CreateOnTransform()
    DO_NOT_CREATE = _DoNotCreate()


class SageMakerEstimatorBase(SageMakerJavaWrapper, JavaEstimator):
    """Adapts a SageMaker learning Algorithm to a Spark Estimator.

    Fits a :class:`~sagemaker_pyspark.SageMakerModel` by running a SageMaker Training Job on a Spark
    Dataset. Each call to :meth:`.fit` submits a new SageMaker Training Job, creates a new
    SageMaker Model, and creates a new SageMaker Endpoint Config. A new Endpoint is either
    created by or the returned SageMakerModel is configured to generate an Endpoint on
    SageMakerModel transform.

    On fit, the input :class:`~pyspark.sql.Dataset` is serialized with the specified
    trainingSparkDataFormat using the specified trainingSparkDataFormatOptions and uploaded
    to an S3 location specified by ``trainingInputS3DataPath``. The serialized Dataset
    is compressed with ``trainingCompressionCodec``, if not None.

    ``trainingProjectedColumns`` can be used to control which columns on the input Dataset are
    transmitted to SageMaker. If not None, then only those column names will be serialized as input
    to the SageMaker Training Job.

    A Training Job is created with the uploaded Dataset being input to the specified
    ``trainingChannelName``, with the specified ``trainingInputMode``. The algorithm is
    specified ``trainingImage``, a Docker image URI reference. The Training Job is created with
    trainingInstanceCount instances of type ``trainingInstanceType``. The Training Job will
    time-out after attr:`trainingMaxRuntimeInSeconds`, if not None.

    SageMaker Training Job hyperparameters are built from the :class:`~pyspark.ml.param.Param`s
    set on this Estimator. Param objects set on this Estimator are retrieved during fit and
    converted to a SageMaker Training Job hyperparameter Map. Param objects are iterated over by
    invoking :meth:`pyspark.ml.param.Params.params` on this Estimator.
    Param objects with neither a default value nor a set value are ignored. If a Param is not set
    but has a default value, the default value will be used. Param values are converted to SageMaker
    hyperparameter String values.

    SageMaker uses the IAM Role with ARN ``sagemakerRole`` to access the input and output S3
    buckets and trainingImage if the image is hosted in ECR. SageMaker Training Job output is
    stored in a Training Job specific sub-prefix of ``trainingOutputS3DataPath``. This contains
    the SageMaker Training Job output file as well as the SageMaker Training Job model file.

    After the Training Job is created, this Estimator will poll for success. Upon success a
    SageMakerModel is created and returned from fit. The SageMakerModel is created with a
    modelImage Docker image URI, defining the SageMaker model primary container and with
    ``modelEnvironmentVariables`` environment variables. Each SageMakerModel has a corresponding
    SageMaker hosting Endpoint. This Endpoint runs on at least endpointInitialInstanceCount
    instances of type endpointInstanceType. The Endpoint is created either during construction of
    the SageMakerModel or on the first call to
    :class:`~sagemaker_pyspark.JavaSageMakerModel.transform`, controlled  by
    ``endpointCreationPolicy``.
    Each Endpointinstance runs with sagemakerRole IAMRole.

    The transform method on SageMakerModel uses ``requestRowSerializer`` to serialize Rows from
    the Dataset undergoing transformation, to requests on the hosted SageMaker Endpoint. The
    ``responseRowDeserializer`` is used to convert the response from the Endpoint to a series of
    Rows, forming the transformed Dataset. If ``modelPrependInputRowsToTransformationRows`` is
    true, then each transformed Row is also prepended with its corresponding input Row.
    """
    __metaclass__ = ABCMeta

    @keyword_only
    def __init__(self, **kwargs):

        super(SageMakerEstimatorBase, self).__init__()
        self._java_obj = self._get_java_obj(**kwargs)
        self._resetUid(self._call_java("uid"))

    @abstractmethod
    def _get_java_obj(self, **kwargs):
        raise NotImplementedError()

    @property
    def latestTrainingJob(self):
        return self._call_java("latestTrainingJob")

    @property
    def trainingImage(self):
        return self._call_java("trainingImage")

    @property
    def modelImage(self):
        return self._call_java("modelImage")

    @property
    def requestRowSerializer(self):
        return self._call_java("requestRowSerializer")

    @property
    def responseRowDeserializer(self):
        return self._call_java("responseRowDeserializer")

    @property
    def sagemakerRole(self):
        return self._call_java("sagemakerRole")

    @property
    def trainingInputS3DataPath(self):
        return self._call_java("trainingInputS3DataPath")

    @property
    def trainingOutputS3DataPath(self):
        return self._call_java("trainingOutputS3DataPath")

    @property
    def trainingInstanceType(self):
        return self._call_java("trainingInstanceType")

    @property
    def trainingInstanceCount(self):
        return self._call_java("trainingInstanceCount")

    @property
    def trainingInstanceVolumeSizeInGB(self):
        return self._call_java("trainingInstanceVolumeSizeInGB")

    @property
    def trainingProjectedColumns(self):
        return self._call_java("trainingProjectedColumns")

    @property
    def trainingChannelName(self):
        return self._call_java("trainingChannelName")

    @property
    def trainingContentType(self):
        return self._call_java("trainingContentType")

    @property
    def trainingS3DataDistribution(self):
        return self._call_java("trainingS3DataDistribution")

    @property
    def trainingSparkDataFormat(self):
        return self._call_java("trainingSparkDataFormat")

    @property
    def trainingSparkDataFormatOptions(self):
        return self._call_java("trainingSparkDataFormatOptions")

    @property
    def trainingInputMode(self):
        return self._call_java("trainingInputMode")

    @property
    def trainingCompressionCodec(self):
        return self._call_java("trainingCompressionCodec")

    @property
    def trainingMaxRuntimeInSeconds(self):
        return self._call_java("trainingMaxRuntimeInSeconds")

    @property
    def trainingKmsKeyId(self):
        return self._call_java("trainingKmsKeyId")

    @property
    def modelEnvironmentVariables(self):
        return self._call_java("modelEnvironmentVariables")

    @property
    def endpointInstanceType(self):
        return self._call_java("endpointInstanceType")

    @property
    def endpointInitialInstanceCount(self):
        return self._call_java("endpointInitialInstanceCount")

    @property
    def endpointCreationPolicy(self):
        return self._call_java("endpointCreationPolicy")

    @property
    def sagemakerClient(self):
        return self._call_java("sagemakerClient")

    @property
    def s3Client(self):
        return self._call_java("s3Client")

    @property
    def stsClient(self):
        return self._call_java("stsClient")

    @property
    def modelPrependInputRowsToTransformationRows(self):
        return self._call_java("modelPrependInputRowsToTransformationRows")

    @property
    def deleteStagingDataAfterTraining(self):
        return self._call_java("deleteStagingDataAfterTraining")

    @property
    def namePolicyFactory(self):
        return self._call_java("namePolicyFactory")

    @property
    def hyperParameters(self):
        return self._call_java("hyperParameters")

    def fit(self, dataset):
        """Fits a SageMakerModel on dataset by running a SageMaker training job.

        Args:
            dataset (Dataset): the dataset to use for the training job.

        Returns:
            JavaSageMakerModel: The Model created by the training job.
        """
        self._transfer_params_to_java()
        return self._call_java("fit", dataset)

    def copy(self, extra):
        raise NotImplementedError()

    # Create model is a no-op for us since we override fit().
    def _create_model(self, java_model):
        pass

    def _to_java(self):
        self._transfer_params_to_java()
        return self._java_object


class SageMakerEstimator(SageMakerEstimatorBase):
    """Adapts a SageMaker learning Algorithm to a Spark Estimator.

    Fits a :class:`~sagemaker_pyspark.SageMakerModel` by running a SageMaker Training Job on a Spark
    Dataset. Each call to :meth:`.fit` submits a new SageMaker Training Job, creates a new
    SageMaker Model, and creates a new SageMaker Endpoint Config. A new Endpoint is either
    created by or the returned SageMakerModel is configured to generate an Endpoint on
    SageMakerModel transform.

    On fit, the input :class:`~pyspark.sql.Dataset` is serialized with the specified
    trainingSparkDataFormat using the specified trainingSparkDataFormatOptions and uploaded
    to an S3 location specified by ``trainingInputS3DataPath``. The serialized Dataset
    is compressed with ``trainingCompressionCodec``, if not None.

    ``trainingProjectedColumns`` can be used to control which columns on the input Dataset are
    transmitted to SageMaker. If not None, then only those column names will be serialized as input
    to the SageMaker Training Job.

    A Training Job is created with the uploaded Dataset being input to the specified
    ``trainingChannelName``, with the specified ``trainingInputMode``. The algorithm is
    specified ``trainingImage``, a Docker image URI reference. The Training Job is created with
    trainingInstanceCount instances of type ``trainingInstanceType``. The Training Job will
    time-out after attr:`trainingMaxRuntimeInSeconds`, if not None.

    SageMaker Training Job hyperparameters are built from the :class:`~pyspark.ml.param.Param`s
    set on this Estimator. Param objects set on this Estimator are retrieved during fit and
    converted to a SageMaker Training Job hyperparameter Map. Param objects are iterated over by
    invoking :meth:`pyspark.ml.param.Params.params` on this Estimator.
    Param objects with neither a default value nor a set value are ignored. If a Param is not set
    but has a default value, the default value will be used. Param values are converted to SageMaker
    hyperparameter String values.

    SageMaker uses the IAM Role with ARN ``sagemakerRole`` to access the input and output S3
    buckets and trainingImage if the image is hosted in ECR. SageMaker Training Job output is
    stored in a Training Job specific sub-prefix of ``trainingOutputS3DataPath``. This contains
    the SageMaker Training Job output file as well as the SageMaker Training Job model file.

    After the Training Job is created, this Estimator will poll for success. Upon success a
    SageMakerModel is created and returned from fit. The SageMakerModel is created with a
    modelImage Docker image URI, defining the SageMaker model primary container and with
    ``modelEnvironmentVariables`` environment variables. Each SageMakerModel has a corresponding
    SageMaker hosting Endpoint. This Endpoint runs on at least endpointInitialInstanceCount
    instances of type endpointInstanceType. The Endpoint is created either during construction of
    the SageMakerModel or on the first call to
    :class:`~sagemaker_pyspark.JavaSageMakerModel.transform`, controlled  by
    ``endpointCreationPolicy``.
    Each Endpointinstance runs with sagemakerRole IAMRole.

    The transform method on SageMakerModel uses ``requestRowSerializer`` to serialize Rows from
    the Dataset undergoing transformation, to requests on the hosted SageMaker Endpoint. The
    ``responseRowDeserializer`` is used to convert the response from the Endpoint to a series of
    Rows, forming the transformed Dataset. If ``modelPrependInputRowsToTransformationRows`` is
    true, then each transformed Row is also prepended with its corresponding input Row.

    Args:
        trainingImage (String): A SageMaker Training Job Algorithm Specification Training Image
            Docker image URI.
        modelImage (String): A SageMaker Model hosting Docker image URI.
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
        hyperParameters (dict): A dict from hyperParameter names to their respective values for
            training.
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
    _wrapped_class = "com.amazonaws.services.sagemaker.sparksdk.SageMakerEstimator"

    def __init__(self,
                 trainingImage,
                 modelImage,
                 trainingInstanceType,
                 trainingInstanceCount,
                 endpointInstanceType,
                 endpointInitialInstanceCount,
                 requestRowSerializer,
                 responseRowDeserializer,
                 hyperParameters=None,
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
                 sagemakerRole=IAMRoleFromConfig(),
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

        if hyperParameters is None:
            hyperParameters = {}

        if uid is None:
            uid = Identifiable._randomUID()

        kwargs = locals().copy()
        del kwargs['self']

        super(SageMakerEstimator, self).__init__(**kwargs)

    def _get_java_obj(self, **kwargs):

        return self._new_java_obj(
            SageMakerEstimator._wrapped_class,
            kwargs['trainingImage'],
            kwargs['modelImage'],
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
            kwargs['s3Client'],
            kwargs['stsClient'],
            kwargs['modelPrependInputRowsToTransformationRows'],
            kwargs['deleteStagingDataAfterTraining'],
            kwargs['namePolicyFactory'],
            kwargs['uid'],
            kwargs['hyperParameters']
        )
