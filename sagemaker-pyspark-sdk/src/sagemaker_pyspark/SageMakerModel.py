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

from pyspark import keyword_only
from pyspark.ml.util import Identifiable
from pyspark.ml.wrapper import JavaModel

from sagemaker_pyspark import (SageMakerJavaWrapper, Option, EndpointCreationPolicy,
                               RandomNamePolicy, SageMakerClients)


class SageMakerModel(SageMakerJavaWrapper, JavaModel):
    """
    A Model implementation which transforms a DataFrame by making requests to a SageMaker Endpoint.
    Manages life cycle of all necessary SageMaker entities, including Model, EndpointConfig,
    and Endpoint.

    This Model transforms one DataFrame to another by repeated, distributed SageMaker Endpoint
    invocation.
    Each invocation request body is formed by concatenating input DataFrame Rows serialized to
    Byte Arrays by the specified
    :class:`~sagemaker_pyspark.transformation.serializers.RequestRowSerializer`. The
    invocation request content-type property is set from
    :attr:`RequestRowSerializer.contentType`. The invocation request accepts property is set
    from :attr:`ResponseRowDeserializer.accepts`.

    The transformed DataFrame is produced by deserializing each invocation response body into a
    series of Rows. Row deserialization is delegated to the specified
    :class:`~sagemaker_pyspark.transformation.deserializers.ResponseRowDeserializer`. If
    prependInputRows is false, the transformed DataFrame
    will contain just these Rows. If prependInputRows is true, then each transformed Row is a
    concatenation of the input Row with its corresponding SageMaker invocation deserialized Row.

    Each invocation of :meth:`~sagemaker_pyspark.JavaSageMakerModel.transform` passes the
    :attr:`Dataset.schema` of the input DataFrame to requestRowSerialize by invoking
    :meth:`RequestRowSerializer.setSchema`.

    The specified RequestRowSerializer also controls the validity of input Row Schemas for
    this Model. Schema validation is carried out on each call to
    :meth:`~sagemaker_pyspark.JavaSageMakerModel.transformSchema`, which invokes
    :meth:`RequestRowSerializer.validateSchema`.

    Adapting this SageMaker model to the data format and type of a specific Endpoint is achieved by
    sub-classing RequestRowSerializer and ResponseRowDeserializer.
    Examples of a Serializer and Deseralizer are
    :class:`~sagemaker_pyspark.transformation.serializers.LibSVMRequestRowSerializer` and
    :class:`~sagemaker_pyspark.transformation.deserializers.LibSVMResponseRowDeserializer`
    respectively.

    Args:
        endpointInstanceType (str): The instance type used to run the model container
        endpointInitialInstanceCount (int): The initial number of instances used to host the model
        requestRowSerializer (RequestRowSerializer): Serializes a Row to an Array of Bytes
        responseRowDeserializer (ResponseRowDeserializer): Deserializes an Array of Bytes to a
            series of Rows
        existingEndpointName (str): An endpoint name
        modelImage (str): A Docker image URI
        modelPath (str): An S3 location that a successfully completed SageMaker Training Job has
            stored its model output to.
        modelEnvironmentVariables (dict): The environment variables that SageMaker will set on the
            model container during execution.
        modelExecutionRoleARN (str): The IAM Role used by SageMaker when running the hosted Model
            and to download model data from S3
        endpointCreationPolicy (EndpointCreationPolicy): Whether the endpoint is created upon
            SageMakerModel construction, transformation, or not at all.
        sagemakerClient (AmazonSageMaker) Amazon SageMaker client. Used to send CreateTrainingJob,
            CreateModel, and CreateEndpoint requests.
        prependResultRows (bool): Whether the transformation result should also include the input
            Rows. If true, each output Row is formed by a concatenation of the input Row with the
            corresponding Row produced by SageMaker invocation, produced by responseRowDeserializer.
            If false, each output Row is just taken from responseRowDeserializer.
        namePolicy (NamePolicy): The NamePolicy to use when naming SageMaker entities created during
            usage of this Model.
        uid (str): The unique identifier of this Estimator. Used to represent this stage in Spark ML
            pipelines.
    """

    _wrapped_class = "com.amazonaws.services.sagemaker.sparksdk.SageMakerModel"

    @keyword_only
    def __init__(self,
                 endpointInstanceType,
                 endpointInitialInstanceCount,
                 requestRowSerializer,
                 responseRowDeserializer,
                 existingEndpointName=None,
                 modelImage=None,
                 modelPath=None,
                 modelEnvironmentVariables=None,
                 modelExecutionRoleARN=None,
                 endpointCreationPolicy=EndpointCreationPolicy.CREATE_ON_CONSTRUCT,
                 sagemakerClient=SageMakerClients.create_sagemaker_client(),
                 prependResultRows=True,
                 namePolicy=RandomNamePolicy(),
                 uid=None,
                 javaObject=None):

        super(SageMakerModel, self).__init__()

        if modelEnvironmentVariables is None:
            modelEnvironmentVariables = {}

        if javaObject:
            self._java_obj = javaObject
        else:
            if uid is None:
                uid = Identifiable._randomUID()

            self._java_obj = self._new_java_obj(
                SageMakerModel._wrapped_class,
                Option(endpointInstanceType),
                Option(endpointInitialInstanceCount),
                requestRowSerializer,
                responseRowDeserializer,
                Option(existingEndpointName),
                Option(modelImage),
                Option(modelPath),
                modelEnvironmentVariables,
                Option(modelExecutionRoleARN),
                endpointCreationPolicy,
                sagemakerClient,
                prependResultRows,
                namePolicy,
                uid
            )
        self._resetUid(self._call_java("uid"))

    @classmethod
    def fromTrainingJob(cls,
                        trainingJobName,
                        modelImage,
                        modelExecutionRoleARN,
                        endpointInstanceType,
                        endpointInitialInstanceCount,
                        requestRowSerializer,
                        responseRowDeserializer,
                        modelEnvironmentVariables=None,
                        endpointCreationPolicy=EndpointCreationPolicy.CREATE_ON_CONSTRUCT,
                        sagemakerClient=SageMakerClients.create_sagemaker_client(),
                        prependResultRows=True,
                        namePolicy=RandomNamePolicy(),
                        uid="sagemaker"):

        """ Creates a JavaSageMakerModel from a successfully completed training job name.

        The returned JavaSageMakerModel can be used to transform DataFrames.

        Args:
            trainingJobName (str):  Name of the successfully completed training job.
            modelImage (str): URI of the image that will serve model inferences.
            modelExecutionRoleARN (str): The IAM Role used by SageMaker when running the hosted
                Model and to download model data from S3.
            endpointInstanceType (str): The instance type used to run the model container.
            endpointInitialInstanceCount (int): The initial number of instances used to host the
                 model.
            requestRowSerializer (RequestRowSerializer): Serializes a row to an array of bytes.
            responseRowDeserializer (ResponseRowDeserializer): Deserializes an array of bytes to a
                series of rows.
            modelEnvironmentVariables: The environment variables that SageMaker will set on the
                model container during execution.
            endpointCreationPolicy (EndpointCreationPolicy): Whether the endpoint is created upon
                SageMakerModel construction, transformation, or not at all.
            sagemakerClient (AmazonSageMaker) Amazon SageMaker client. Used to send
                CreateTrainingJob, CreateModel, and CreateEndpoint requests.
            prependResultRows (bool): Whether the transformation result should also include the
                input Rows. If true, each output Row is formed by a concatenation of the input Row
                with the corresponding Row produced by SageMaker invocation, produced by
                responseRowDeserializer. If false, each output Row is just taken from
                responseRowDeserializer.
            namePolicy (NamePolicy): The NamePolicy to use when naming SageMaker entities created
                during usage of the returned model.
            uid (String): The unique identifier of the SageMakerModel. Used to represent the stage
                in Spark ML pipelines.

        Returns:
            JavaSageMakerModel: a JavaSageMakerModel that sends InvokeEndpoint requests to an
                endpoint hosting the training job's model.


        """
        scala_function = "%s.fromTrainingJob" % SageMakerModel._wrapped_class

        if modelEnvironmentVariables is None:
            modelEnvironmentVariables = {}

        model_java_obj = SageMakerJavaWrapper()._new_java_obj(
            scala_function,
            trainingJobName,
            modelImage,
            modelExecutionRoleARN,
            endpointInstanceType,
            endpointInitialInstanceCount,
            requestRowSerializer,
            responseRowDeserializer,
            modelEnvironmentVariables,
            endpointCreationPolicy,
            sagemakerClient,
            prependResultRows,
            namePolicy,
            uid)

        return SageMakerModel(
            endpointInstanceType=endpointInstanceType,
            endpointInitialInstanceCount=endpointInitialInstanceCount,
            requestRowSerializer=requestRowSerializer,
            responseRowDeserializer=responseRowDeserializer,
            javaObject=model_java_obj)

    @classmethod
    def fromEndpoint(cls,
                     endpointName,
                     requestRowSerializer,
                     responseRowDeserializer,
                     modelEnvironmentVariables=None,
                     sagemakerClient=SageMakerClients.create_sagemaker_client(),
                     prependResultRows=True,
                     namePolicy=RandomNamePolicy(),
                     uid="sagemaker"):

        """ Creates a JavaSageMakerModel from existing model data in S3.

        The returned JavaSageMakerModel can be used to transform Dataframes.

        Args:
            endpointName (str): The name of an endpoint that is currently in service.
            requestRowSerializer (RequestRowSerializer): Serializes a row to an array of bytes.
            responseRowDeserializer (ResponseRowDeserializer): Deserializes an array of bytes to a
                series of rows.
            modelEnvironmentVariables: The environment variables that SageMaker will set on the
                model container during execution.
            sagemakerClient (AmazonSageMaker) Amazon SageMaker client. Used to send
                CreateTrainingJob, CreateModel, and CreateEndpoint requests.
            prependResultRows (bool): Whether the transformation result should also include the
                input Rows. If true, each output Row is formed by a concatenation of the input Row
                with the corresponding Row produced by SageMaker invocation, produced by
                responseRowDeserializer. If false, each output Row is just taken from
                responseRowDeserializer.
            namePolicy (NamePolicy): The NamePolicy to use when naming SageMaker entities created
                during usage of the returned model.
            uid (String): The unique identifier of the SageMakerModel. Used to represent the stage
                in Spark ML pipelines.

        Returns:
            JavaSageMakerModel:
                A JavaSageMakerModel that sends InvokeEndpoint requests to an endpoint hosting
                the training job's model.

        """

        scala_function = "%s.fromEndpoint" % SageMakerModel._wrapped_class

        if modelEnvironmentVariables is None:
            modelEnvironmentVariables = {}

        model_java_obj = SageMakerJavaWrapper()._new_java_obj(
            scala_function,
            endpointName,
            requestRowSerializer,
            responseRowDeserializer,
            modelEnvironmentVariables,
            sagemakerClient,
            prependResultRows,
            namePolicy,
            uid)

        return SageMakerModel(
            endpointInstanceType=None,
            endpointInitialInstanceCount=None,
            requestRowSerializer=requestRowSerializer,
            responseRowDeserializer=responseRowDeserializer,
            javaObject=model_java_obj)

    @classmethod
    def fromModelS3Path(cls,
                        modelPath,
                        modelImage,
                        modelExecutionRoleARN,
                        endpointInstanceType,
                        endpointInitialInstanceCount,
                        requestRowSerializer,
                        responseRowDeserializer,
                        modelEnvironmentVariables=None,
                        endpointCreationPolicy=EndpointCreationPolicy.CREATE_ON_CONSTRUCT,
                        sagemakerClient=SageMakerClients.create_sagemaker_client(),
                        prependResultRows=True,
                        namePolicy=RandomNamePolicy(),
                        uid="sagemaker"):

        """ Creates a JavaSageMakerModel from existing model data in S3.

        The returned JavaSageMakerModel can be used to transform Dataframes.

        Args:
            modelPath (str): The S3 URI to the model  data to host.
            modelImage (str): The URI of the image that will serve model inferences.
            modelExecutionRoleARN (str): The IAM Role used by SageMaker when running the hosted
                Model and to download model data from S3.
            endpointInstanceType (str): The instance type used to run the model container.
            endpointInitialInstanceCount (int): The initial number of instances used to host the
                model.
            requestRowSerializer (RequestRowSerializer): Serializes a row to an array of bytes.
            responseRowDeserializer (ResponseRowDeserializer): Deserializes an array of bytes to a
                series of rows.
            modelEnvironmentVariables: The environment variables that SageMaker will set on the
                model container during execution.
            endpointCreationPolicy (EndpointCreationPolicy): Whether the endpoint is created upon
                SageMakerModel construction, transformation, or not at all.
            sagemakerClient (AmazonSageMaker) Amazon SageMaker client. Used to send
                CreateTrainingJob, CreateModel, and CreateEndpoint requests.
            prependResultRows (bool): Whether the transformation result should also include the
                input Rows. If true, each output Row is formed by a concatenation of the input Row
                with the corresponding Row produced by SageMaker invocation, produced by
                responseRowDeserializer. If false, each output Row is just taken from
                responseRowDeserializer.
            namePolicy (NamePolicy): The NamePolicy to use when naming SageMaker entities created
                during usage of the returned model.
            uid (String): The unique identifier of the SageMakerModel. Used to represent the stage
                in Spark ML pipelines.

        Returns:
            JavaSageMakerModel:
                A JavaSageMakerModel that sends InvokeEndpoint requests to an endpoint hosting
                the training job's model.

        """

        scala_function = "%s.fromModelS3Path" % SageMakerModel._wrapped_class

        if modelEnvironmentVariables is None:
            modelEnvironmentVariables = {}

        model_java_obj = SageMakerJavaWrapper()._new_java_obj(
            scala_function,
            modelPath,
            modelImage,
            modelExecutionRoleARN,
            endpointInstanceType,
            endpointInitialInstanceCount,
            requestRowSerializer,
            responseRowDeserializer,
            modelEnvironmentVariables,
            endpointCreationPolicy,
            sagemakerClient,
            prependResultRows,
            namePolicy,
            uid)

        return SageMakerModel(
            endpointInstanceType=endpointInstanceType,
            endpointInitialInstanceCount=endpointInitialInstanceCount,
            requestRowSerializer=requestRowSerializer,
            responseRowDeserializer=responseRowDeserializer,
            javaObject=model_java_obj)

    @property
    def endpointInstanceType(self):
        return self._call_java("endpointInstanceType")

    @property
    def endpointInitialInstanceCount(self):
        return self._call_java("endpointInitialInstanceCount")

    @property
    def requestRowSerializer(self):
        return self._call_java("requestRowSerializer")

    @property
    def responseRowDeserializer(self):
        return self._call_java("responseRowDeserializer")

    @property
    def existingEndpointName(self):
        return self._call_java("existingEndpointName")

    @property
    def modelImage(self):
        return self._call_java("modelImage")

    @property
    def modelPath(self):
        return self._call_java("modelPath")

    @property
    def modelEnvironmentVariables(self):
        return self._call_java("modelEnvironmentVariables")

    @property
    def modelExecutionRoleARN(self):
        return self._call_java("modelExecutionRoleARN")

    @property
    def sagemakerClient(self):
        return self._call_java("sagemakerClient")

    @property
    def endpointCreationPolicy(self):
        return self._call_java("endpointCreationPolicy")

    @property
    def prependResultRows(self):
        return self._call_java("prependResultRows")

    @property
    def namePolicy(self):
        return self._call_java("namePolicy")

    @property
    def endpointName(self):
        return self._call_java("endpointName")

    @property
    def resourceCleanup(self):
        return self._call_java("resourceCleanup")

    def getCreatedResources(self):
        return self._call_java("getCreatedResources")

    def transform(self, dataset):
        return self._call_java("transform", dataset)

    def transformSchema(self, schema):
        return self._call_java("transformSchema", schema)

    def _to_java(self):
        return self._java_obj

    @classmethod
    def _from_java(cls, JavaObject):
        return SageMakerModel(endpointInstanceType=None,
                              endpointInitialInstanceCount=None,
                              requestRowSerializer=None,
                              responseRowDeserializer=None,
                              javaObject=JavaObject)

    def _transform(self, dataset):
        pass
