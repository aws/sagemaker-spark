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

package com.amazonaws.services.sagemaker.sparksdk

import org.mockito.ArgumentCaptor
import org.mockito.Matchers.any
import org.mockito.Mockito.inOrder
import org.mockito.Mockito.never
import org.mockito.Mockito.times
import org.mockito.Mockito.verify
import org.mockito.Mockito.when
import org.scalatest.BeforeAndAfter
import org.scalatest.FlatSpec
import org.scalatest.mockito.MockitoSugar
import scala.jdk.CollectionConverters._

import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql._
import org.apache.spark.sql.types._

import com.amazonaws.services.sagemaker.AmazonSageMaker
import com.amazonaws.services.sagemaker.model._
import com.amazonaws.services.sagemaker.sparksdk.EndpointCreationPolicy.EndpointCreationPolicy
import com.amazonaws.services.sagemaker.sparksdk.exceptions.SageMakerSparkSDKException
import com.amazonaws.services.sagemaker.sparksdk.internal.TimeProvider
import com.amazonaws.services.sagemaker.sparksdk.transformation.{RequestRowSerializer, ResponseRowDeserializer}
import com.amazonaws.services.sagemaker.sparksdk.transformation.deserializers.LibSVMResponseRowDeserializer
import com.amazonaws.services.sagemaker.sparksdk.transformation.serializers.{ProtobufRequestRowSerializer, UnlabeledCSVRequestRowSerializer}

class SageMakerModelTests extends FlatSpec with MockitoSugar
  with BeforeAndAfter {

  var sagemakerMock: AmazonSageMaker = _
  var requestRowSerializerMock: RequestRowSerializer = _
  var responseRowDeserializerMock: ResponseRowDeserializer = _
  var dataSetMock : Dataset[Row] = _
  var sparkSession : SparkSession = _

  val namePolicy = RandomNamePolicy("Elizabeth")

  before {
    sagemakerMock = mock[AmazonSageMaker]
    requestRowSerializerMock = mock[RequestRowSerializer]
    responseRowDeserializerMock = mock[ResponseRowDeserializer]
    when(responseRowDeserializerMock.schema).thenReturn(new StructType(Array(
      StructField("a", IntegerType))))
    dataSetMock = mock[DataFrame]
    when(dataSetMock.toDF()).thenReturn(dataSetMock)
    when(dataSetMock.mapPartitions(any(classOf[Function1[Iterator[Row], Iterator[Row]]]))
      (any[Encoder[Row]])).thenReturn(dataSetMock)
  }

  def dummyModel(endpointName: Option[String] = Option.empty,
                 modelImage : Option[String] = Some("dummy-image"),
                 modelPath : Option[S3DataPath] = Some(S3DataPath ("a", "b")),
                 requestRowSerializer: RequestRowSerializer = requestRowSerializerMock,
                 responseRowDeserializer: ResponseRowDeserializer = responseRowDeserializerMock,
                 modelEnvironmentVariables : Map[String, String] = Map(),
                 modelExecutionRoleARN : Option[String] = Some("role"),
                 endpointCreationPolicy : EndpointCreationPolicy =
                   EndpointCreationPolicy.CREATE_ON_CONSTRUCT,
                 endpointInstanceType : String = "c4.2048xlarge",
                 endpointInitialInstanceCount : Int = 1,
                 prependResultRows : Boolean = false,
                 namePolicy : NamePolicy = namePolicy,
                 uid : String = "uid") : SageMakerModel = {
    new SageMakerModel(Some(endpointInstanceType), Some(endpointInitialInstanceCount),
      requestRowSerializer, responseRowDeserializer, endpointName, modelImage, modelPath,
      modelEnvironmentVariables, modelExecutionRoleARN, endpointCreationPolicy, sagemakerMock,
      prependResultRows, namePolicy, uid)
  }

  "SageMakerModel" should "create hosting resources on startup by default in order, and not " +
    "delete them" in {
    val model = dummyModel(endpointCreationPolicy = EndpointCreationPolicy.CREATE_ON_TRANSFORM)
    setupDescribeEndpointResponses(EndpointStatus.InService)

    model.transform(dataSetMock)
    val order = inOrder(sagemakerMock, dataSetMock)
    order.verify(sagemakerMock).createModel(any[CreateModelRequest])
    order.verify(sagemakerMock).createEndpointConfig(any[CreateEndpointConfigRequest])
    order.verify(sagemakerMock).createEndpoint(any[CreateEndpointRequest])
    order.verify(sagemakerMock).describeEndpoint(any[DescribeEndpointRequest])

    order.verify(sagemakerMock, never).deleteEndpoint(any[DeleteEndpointRequest])
    order.verify(sagemakerMock, never).deleteEndpointConfig(any[DeleteEndpointConfigRequest])
    order.verify(sagemakerMock, never).deleteModel(any[DeleteModelRequest])
  }

  it should "construct a copy" in {
    val model = dummyModel(endpointCreationPolicy = EndpointCreationPolicy.CREATE_ON_TRANSFORM)
    val copiedModel = model.copy(new ParamMap())
    assert(copiedModel.endpointInstanceType == model.endpointInstanceType)
    assert(copiedModel.endpointInitialInstanceCount == model.endpointInitialInstanceCount)
    assert(copiedModel.modelImage == model.modelImage)
    assert(copiedModel.existingEndpointName == model.existingEndpointName)
    assert(copiedModel.responseRowDeserializer == model.responseRowDeserializer)
    assert(copiedModel.requestRowSerializer == model.requestRowSerializer)
    assert(copiedModel.modelEnvironmentVariables == model.modelEnvironmentVariables)
    assert(copiedModel.modelExecutionRoleARN == model.modelExecutionRoleARN)
    assert(copiedModel.prependResultRows == model.prependResultRows)
    assert(copiedModel.namePolicy == model.namePolicy)
    assert(copiedModel.sagemakerClient == model.sagemakerClient)
    assert(copiedModel.endpointCreationPolicy == EndpointCreationPolicy.DO_NOT_CREATE)
    assert(copiedModel.uid == model.uid)
  }

  it should "create hosting resources on transform if CREATE_ON_TRANSFORM policy is set" in {
    val model = dummyModel(endpointCreationPolicy = EndpointCreationPolicy.CREATE_ON_TRANSFORM)
    setupDescribeEndpointResponses(EndpointStatus.InService)

    verify(sagemakerMock, never).createModel(any[CreateModelRequest])
    verify(sagemakerMock, never).createEndpointConfig(any[CreateEndpointConfigRequest])
    verify(sagemakerMock, never).createEndpoint(any[CreateEndpointRequest])
    verify(sagemakerMock, never).describeEndpoint(any[DescribeEndpointRequest])
    model.transform(dataSetMock)
    val order = inOrder(sagemakerMock)
    order.verify(sagemakerMock).createModel(any[CreateModelRequest])
    order.verify(sagemakerMock).createEndpointConfig(any[CreateEndpointConfigRequest])
    order.verify(sagemakerMock).createEndpoint(any[CreateEndpointRequest])
    order.verify(sagemakerMock).describeEndpoint(any[DescribeEndpointRequest])
  }

  it should "use correct model name" in {
    setupDescribeEndpointResponses(EndpointStatus.InService)
    val model = dummyModel()

    val captor = ArgumentCaptor.forClass(classOf[CreateModelRequest])
    verify(sagemakerMock).createModel(captor.capture)
    assert(captor.getValue.getModelName == model.namePolicy.modelName)
  }

  it should "use correct name and model parameters" in {


    setupDescribeEndpointResponses(EndpointStatus.InService)
    val model = dummyModel()

    val captor = ArgumentCaptor.forClass(classOf[CreateModelRequest])
    verify(sagemakerMock).createModel(captor.capture)

    // correct name
    assert(captor.getValue.getModelName == model.namePolicy.modelName)
    // correct parameters
    assert(captor.getValue.getExecutionRoleArn == model.modelExecutionRoleARN.get)
    val containerDef = captor.getValue.getPrimaryContainer
    assert(containerDef.getImage == model.modelImage.get)
    assert(containerDef.getEnvironment == model.modelEnvironmentVariables.asJava)
    // no supplemental containers should be set
    assert(captor.getValue.getContainers == null)
  }

  it should "use correct name and endpointConfig parameters" in {


    setupDescribeEndpointResponses(EndpointStatus.InService)
    val model = dummyModel()

    val captor = ArgumentCaptor.forClass(classOf[CreateEndpointConfigRequest])
    verify(sagemakerMock).createEndpointConfig(captor.capture)

    // correct name
    assert(captor.getValue.getEndpointConfigName == model.namePolicy.endpointConfigName)
    // only one variant
    val productionVariants = captor.getValue.getProductionVariants
    assert(productionVariants.size == 1)
    val pv = productionVariants.get(0)
    // correct parameters set
    assert(pv.getInstanceType == model.endpointInstanceType.get)
    assert(pv.getInitialInstanceCount == model.endpointInitialInstanceCount.get)
  }

  it should "use correct endpoint name" in {
    setupDescribeEndpointResponses(EndpointStatus.InService)
    val model = dummyModel()

    val captor = ArgumentCaptor.forClass(classOf[CreateEndpointRequest])
    verify(sagemakerMock).createEndpoint(captor.capture)
    assert(captor.getValue.getEndpointName == model.namePolicy.endpointName)
  }

  it should "poll for endpoint completion" in {
    val timeProviderMock = mock[TimeProvider]
    when(timeProviderMock.currentTimeMillis).thenReturn(0)

    val model = dummyModel(endpointCreationPolicy = EndpointCreationPolicy.CREATE_ON_TRANSFORM)
    model.timeProvider = timeProviderMock

    setupDescribeEndpointResponses(EndpointStatus.Creating, EndpointStatus.Creating,
        EndpointStatus.InService)

    model.transform(dataSetMock)

    verify(sagemakerMock, times(3)).describeEndpoint(any[DescribeEndpointRequest])
    verify(timeProviderMock, times(2)).sleep(SageMakerModel.EndpointPollInterval.toMillis)
  }

  it should "ignore transient failures when polling for endpoint completion" in {
    val timeProviderMock = mock[TimeProvider]
    when(timeProviderMock.currentTimeMillis).thenReturn(0)
    val retryableException = new AmazonSageMakerException("unavailable")
    retryableException.setStatusCode(500)
    when(sagemakerMock.describeEndpoint(any[DescribeEndpointRequest]))
      .thenThrow(retryableException)
      .thenReturn(statusToResult(EndpointStatus.Creating))
      .thenReturn(statusToResult(EndpointStatus.InService))

    val model = dummyModel(endpointCreationPolicy = EndpointCreationPolicy.CREATE_ON_TRANSFORM)
    model.timeProvider = timeProviderMock
    model.transform(dataSetMock)
    verify(sagemakerMock, times(3)).describeEndpoint(any[DescribeEndpointRequest])
    verify(timeProviderMock, times(2)).sleep(SageMakerModel.EndpointPollInterval.toMillis)
  }

  it should "throw SageMakerSparkSDKException if model creation fails" in {
    when(sagemakerMock.createModel(any[CreateModelRequest]))
      .thenThrow(new AmazonSageMakerException("failed"))

    val caught =
      intercept[SageMakerSparkSDKException] {
        dummyModel()
      }

    assert(caught.createdResources.modelName.isEmpty)
    assert(caught.createdResources.endpointConfigName.isEmpty)
    assert(caught.createdResources.endpointName.isEmpty)

    verify(sagemakerMock, never).createEndpointConfig(any[CreateEndpointConfigRequest])
    verify(sagemakerMock, never).createEndpoint(any[CreateEndpointRequest])
  }

  it should "throw SageMakerSparkSDKException if endpointConfig creation fails" in {
    when(sagemakerMock.createEndpointConfig(any[CreateEndpointConfigRequest]))
      .thenThrow(new AmazonSageMakerException("failed"))

    val caught =
      intercept[SageMakerSparkSDKException] {
        dummyModel()
      }

    assert(caught.createdResources.modelName.get == namePolicy.modelName)
    assert(caught.createdResources.endpointConfigName.isEmpty)
    assert(caught.createdResources.endpointName.isEmpty)

    verify(sagemakerMock, never).createEndpoint(any[CreateEndpointRequest])
  }

  it should "throw SageMakerSparkSDKException if endpoint creation fails synchronously" in {
    when(sagemakerMock.createEndpoint(any[CreateEndpointRequest]))
      .thenThrow(new AmazonSageMakerException("failed"))

    val caught =
      intercept[SageMakerSparkSDKException] {
        dummyModel()
      }

    assert(caught.createdResources.modelName.get == namePolicy.modelName)
    assert(caught.createdResources.endpointConfigName.get == namePolicy.endpointConfigName)
    assert(caught.createdResources.endpointName.isEmpty)
  }

  it should "throw SageMakerSparkSDKException if endpoint creation fails asynchronously" in {
    val timeProviderMock = mock[TimeProvider]
    when(timeProviderMock.currentTimeMillis).thenReturn(0)


    setupDescribeEndpointResponses(EndpointStatus.Creating, EndpointStatus.Creating,
        EndpointStatus.Failed)

    val caught =
      intercept[SageMakerSparkSDKException] {
        val model = dummyModel(endpointCreationPolicy = EndpointCreationPolicy.CREATE_ON_TRANSFORM)
        model.timeProvider = timeProviderMock
        model.transform(dataSetMock)
      }

    assert(caught.createdResources.modelName.get == namePolicy.modelName)
    assert(caught.createdResources.endpointConfigName.get == namePolicy.endpointConfigName)
    assert(caught.createdResources.endpointName.get == namePolicy.endpointName)
  }

  it should "throw SageMakerSparkSDKException if polling for endpoint completion times out" in {
    val timeProviderMock = mock[TimeProvider]
    when(timeProviderMock.currentTimeMillis)
      .thenReturn(0) // starting endpoint creation time
      .thenReturn(0) // await endpoint completion start time
      .thenReturn(1) // first while loop check
      .thenReturn(2) // second while loop check
      // third while loop check - should exit loop
      .thenReturn(SageMakerModel.EndpointCreationTimeout.toMillis + 1)

    setupDescribeEndpointResponses(EndpointStatus.Creating)

    val caught =
      intercept[SageMakerSparkSDKException] {
        val model = dummyModel(endpointCreationPolicy = EndpointCreationPolicy.CREATE_ON_TRANSFORM)
        model.timeProvider = timeProviderMock
        model.transform(dataSetMock)
      }

    assert(caught.createdResources.modelName.get == namePolicy.modelName)
    assert(caught.createdResources.endpointConfigName.get == namePolicy.endpointConfigName)
    assert(caught.createdResources.endpointName.get == namePolicy.endpointName)
    verify(sagemakerMock, times(2)).describeEndpoint(any[DescribeEndpointRequest])
    verify(timeProviderMock, times(2)).sleep(SageMakerModel.EndpointPollInterval.toMillis)
  }

  it should "transform schema" in {
    val outputSchema = StructType(Array(StructField("string", StringType)))
    val inputSchema = StructType(Array(StructField("double", DoubleType)))

    when(responseRowDeserializerMock.schema).thenReturn(outputSchema)
    val model = dummyModel(endpointCreationPolicy = EndpointCreationPolicy.CREATE_ON_TRANSFORM)
    setupDescribeEndpointResponses(EndpointStatus.InService)

    assert(model.transformSchema(inputSchema) == outputSchema)

    val prependingModel = dummyModel(prependResultRows = true)

    assert(prependingModel.transformSchema(inputSchema) == StructType(
      inputSchema.fields ++ outputSchema.fields))
  }

  it should "create a SageMakerModel from a training job name" in {
    val trainingJobName = "endpointName"
    val roleArn = "roleArn"
    val image = "EcrImage"
    val serializer = new UnlabeledCSVRequestRowSerializer()
    val deserializer = new LibSVMResponseRowDeserializer(50)
    val modelEnvironmentVariables = Map[String, String]("key" -> "value")
    val endpointInstanceType = "ml.c4.xlarge"
    val endpointInitialInstanceCount = 2
    val endpointCreationPolicy = EndpointCreationPolicy.CREATE_ON_TRANSFORM
    val prependResultRows = false
    val namePolicy = new RandomNamePolicy()
    val uid = "another uid"

    val modelArtifacts = "s3://model/artifacts"
    val trainingJobStatus = TrainingJobStatus.Completed.toString
    when(sagemakerMock.describeTrainingJob(any[DescribeTrainingJobRequest])).thenReturn(
      new DescribeTrainingJobResult().withTrainingJobStatus(trainingJobStatus).withModelArtifacts(
        new ModelArtifacts().withS3ModelArtifacts(modelArtifacts)))
    val sagemakerModel = SageMakerModel.fromTrainingJob(
      trainingJobName, image, roleArn, endpointInstanceType,
      endpointInitialInstanceCount, serializer, deserializer,
      modelEnvironmentVariables, endpointCreationPolicy, sagemakerMock,
      prependResultRows, namePolicy, uid)

    assert(sagemakerModel.modelPath.get.toS3UriString == modelArtifacts)
    assert(sagemakerModel.modelExecutionRoleARN.get == roleArn)
    assert(sagemakerModel.modelImage.get == image)
    assert(sagemakerModel.requestRowSerializer == serializer)
    assert(sagemakerModel.responseRowDeserializer == deserializer)
    assert(sagemakerModel.modelEnvironmentVariables == modelEnvironmentVariables)
    assert(sagemakerModel.endpointCreationPolicy == endpointCreationPolicy)
    assert(sagemakerModel.endpointInstanceType.get == endpointInstanceType)
    assert(sagemakerModel.endpointInitialInstanceCount.get == endpointInitialInstanceCount)
    assert(sagemakerModel.prependResultRows == prependResultRows)
    assert(sagemakerModel.namePolicy == namePolicy)
    assert(sagemakerModel.uid == uid)
  }

  it should "refuse to create a SageMakerModel from a training job name if the training job " +
    "didn't stop or complete" in {
    val trainingJobName = "endpointName"
    val roleArn = "roleArn"
    val image = "EcrImage"
    val modelArtifacts = "s3://model/artifacts"
    val trainingJobStatus = TrainingJobStatus.Failed.toString
    when(sagemakerMock.describeTrainingJob(any[DescribeTrainingJobRequest])).thenReturn(
      new DescribeTrainingJobResult().withTrainingJobStatus(trainingJobStatus).withModelArtifacts(
        new ModelArtifacts().withS3ModelArtifacts(modelArtifacts)))
    intercept[IllegalArgumentException] {
      SageMakerModel.fromTrainingJob(trainingJobName, image,
        endpointInstanceType = "ml.c4.xlarge",
        endpointInitialInstanceCount = 1,
        requestRowSerializer = new ProtobufRequestRowSerializer(),
        responseRowDeserializer = new LibSVMResponseRowDeserializer(50),
        modelExecutionRoleARN = roleArn,
        sagemakerClient = sagemakerMock)
    }
  }

  it should "refuse to create a SageMakerModel from a training job name if the " +
    "EndpointCreationPolicy is DO_NOT_CREATE" in {
    val trainingJobName = "endpointName"
    val roleArn = "roleArn"
    val image = "EcrImage"
    intercept[IllegalArgumentException] {
      SageMakerModel.fromTrainingJob(trainingJobName, image,
        endpointInstanceType = "ml.c4.xlarge",
        endpointInitialInstanceCount = 1,
        endpointCreationPolicy = EndpointCreationPolicy.DO_NOT_CREATE,
        requestRowSerializer = new ProtobufRequestRowSerializer(),
        responseRowDeserializer = new LibSVMResponseRowDeserializer(50),
        modelExecutionRoleARN = roleArn)
    }
  }

  it should "refuse to create a SageMakerModel from an endpoint name if the endpoint is not " +
    "IN_SERVICE" in {
    val trainingJobName = "endpointName"
    val roleArn = "roleArn"
    val image = "EcrImage"
    setupDescribeEndpointResponses(EndpointStatus.Creating)
    intercept[IllegalArgumentException] {
      SageMakerModel.fromTrainingJob(trainingJobName,
        image,
        roleArn,
        "ml.c4.xlarge",
        1,
        endpointCreationPolicy = EndpointCreationPolicy.DO_NOT_CREATE,
        requestRowSerializer = new ProtobufRequestRowSerializer(),
        responseRowDeserializer = new LibSVMResponseRowDeserializer(50),
        sagemakerClient = sagemakerMock)
    }
  }

  it should "create a SageMakerModel from an endpoint name" in {
    val endpointName = "endpointName"
    val roleArn = "roleArn"
    val serializer = new UnlabeledCSVRequestRowSerializer()
    val deserializer = new LibSVMResponseRowDeserializer(50)
    val modelEnvironmentVariables = Map[String, String]("key" -> "value")
    val endpointInstanceType = "ml.c4.xlarge"
    val endpointInitialInstanceCount = 2
    val prependResultRows = false
    val namePolicy = new RandomNamePolicy()
    val uid = "another uid"
    setupDescribeEndpointResponses(EndpointStatus.InService)
    val sagemakerModel = SageMakerModel.fromEndpoint(
      endpointName,
      requestRowSerializer = serializer,
      responseRowDeserializer = deserializer,
      modelEnvironmentVariables = modelEnvironmentVariables,
      prependResultRows = prependResultRows,
      sagemakerClient = sagemakerMock,
      namePolicy = namePolicy,
      uid = uid)

    assert(sagemakerModel.modelPath.isEmpty)
    assert(sagemakerModel.modelExecutionRoleARN == None)
    assert(sagemakerModel.modelImage.isEmpty)
    assert(sagemakerModel.requestRowSerializer == serializer)
    assert(sagemakerModel.responseRowDeserializer == deserializer)
    assert(sagemakerModel.modelEnvironmentVariables == modelEnvironmentVariables)
    assert(sagemakerModel.endpointCreationPolicy == EndpointCreationPolicy.DO_NOT_CREATE)
    assert(sagemakerModel.endpointInstanceType == None)
    assert(sagemakerModel.endpointInitialInstanceCount == None)
    assert(sagemakerModel.prependResultRows == prependResultRows)
    assert(sagemakerModel.namePolicy == namePolicy)
    assert(sagemakerModel.uid == uid)
    assert(sagemakerModel.endpointName.get == endpointName)
    assert(sagemakerModel.sagemakerClient == sagemakerMock)
  }

  it should "create a model from a model path" in {
    val modelPath = "s3://model/path"
    val roleArn = "roleArn"
    val image = "ECRRepo"
    val serializer = new UnlabeledCSVRequestRowSerializer()
    val deserializer = new LibSVMResponseRowDeserializer(50)
    val modelEnvironmentVariables = Map[String, String]("key" -> "value")
    val endpointCreationPolicy = EndpointCreationPolicy.CREATE_ON_TRANSFORM
    val endpointInstanceType = "ml.c4.xlarge"
    val endpointInitialInstanceCount = 2
    val prependResultRows = false
    val namePolicy = new RandomNamePolicy()
    val uid = "another uid"
    val sagemakerModel = SageMakerModel.fromModelS3Path(modelPath, image,
      roleArn, endpointInstanceType, endpointInitialInstanceCount,
      serializer, deserializer,
      modelEnvironmentVariables, endpointCreationPolicy, sagemakerMock,
      prependResultRows, namePolicy, uid)

    assert(sagemakerModel.modelPath.get.toS3UriString == modelPath)
    assert(sagemakerModel.modelExecutionRoleARN.get == roleArn)
    assert(sagemakerModel.modelImage.get == image)
    assert(sagemakerModel.requestRowSerializer == serializer)
    assert(sagemakerModel.responseRowDeserializer == deserializer)
    assert(sagemakerModel.modelEnvironmentVariables == modelEnvironmentVariables)
    assert(sagemakerModel.endpointCreationPolicy == endpointCreationPolicy)
    assert(sagemakerModel.endpointInstanceType.get == endpointInstanceType)
    assert(sagemakerModel.endpointInitialInstanceCount.get == endpointInitialInstanceCount)
    assert(sagemakerModel.prependResultRows == prependResultRows)
    assert(sagemakerModel.namePolicy == namePolicy)
    assert(sagemakerModel.uid == uid)
    assert(sagemakerModel.sagemakerClient == sagemakerMock)
  }

  it should "refuse to create a model from a model path if the EndpointCreationPolicy is" +
    " DO_NOT_CREATE" in {
    val modelPath = "s3://model/path"
    val roleArn = "roleArn"
    val image = "ECRRepo"
    intercept[IllegalArgumentException] {
      SageMakerModel.fromModelS3Path(
        modelPath = modelPath,
        modelImage = image,
        modelExecutionRoleARN = roleArn,
        endpointInstanceType = "ml.c4.xlarge",
        endpointInitialInstanceCount = 1,
        endpointCreationPolicy = EndpointCreationPolicy.DO_NOT_CREATE,
        requestRowSerializer = new ProtobufRequestRowSerializer(),
        responseRowDeserializer = new LibSVMResponseRowDeserializer(50),
        sagemakerClient = sagemakerMock
        )
    }

  }


  private def statusToResult(status : EndpointStatus): DescribeEndpointResult = {
    new DescribeEndpointResult().withEndpointStatus(status.toString)
  }

  private def statusToResultWithName(namePolicy : NamePolicy) = {
    (status : EndpointStatus) => {
      new DescribeEndpointResult()
          .withEndpointName(namePolicy.endpointName)
          .withEndpointStatus(status.toString)
    }
  }

  private def setupDescribeEndpointResponses(firstStatus: EndpointStatus,
                                             moreStatuses : EndpointStatus*) = {
    when(sagemakerMock.describeEndpoint(any[DescribeEndpointRequest])).thenReturn(
      statusToResultWithName(namePolicy)(firstStatus),
      moreStatuses.map(statusToResultWithName(namePolicy)) : _*)
  }
}
