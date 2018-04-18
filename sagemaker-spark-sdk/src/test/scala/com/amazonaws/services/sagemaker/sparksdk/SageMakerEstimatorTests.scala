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

import java.util

import collection.JavaConverters.mapAsJavaMapConverter
import com.amazonaws.SdkClientException
import com.amazonaws.services.s3.AmazonS3
import com.amazonaws.services.s3.model.{AmazonS3Exception, ObjectListing, S3ObjectSummary}
import com.amazonaws.services.securitytoken.AWSSecurityTokenService
import com.amazonaws.services.securitytoken.model.{GetCallerIdentityRequest, GetCallerIdentityResult}
import org.mockito.ArgumentCaptor
import org.mockito.Matchers.any
import org.mockito.Mockito._
import org.scalatest._
import org.scalatest.mockito.MockitoSugar

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.param.{BooleanParam, IntParam, Param}
import org.apache.spark.sql._

import com.amazonaws.services.sagemaker.AmazonSageMaker
import com.amazonaws.services.sagemaker.model._
import com.amazonaws.services.sagemaker.sparksdk.EndpointCreationPolicy.EndpointCreationPolicy
import com.amazonaws.services.sagemaker.sparksdk.internal.{DataUploader, ManifestDataUploadResult, ObjectPrefixUploadResult, TimeProvider}
import com.amazonaws.services.sagemaker.sparksdk.transformation.{RequestRowSerializer, ResponseRowDeserializer}
import com.amazonaws.services.sagemaker.sparksdk.transformation.deserializers.LibSVMResponseRowDeserializer
import com.amazonaws.services.sagemaker.sparksdk.transformation.serializers.LibSVMRequestRowSerializer

class SageMakerEstimatorTests extends FlatSpec with Matchers with MockitoSugar with BeforeAndAfter {

  var dataset: Dataset[String] = _
  var dataUploaderMock: DataUploader = _
  var timeProviderMock: TimeProvider = _
  var sagemakerMock: AmazonSageMaker = _

  var s3Mock : AmazonS3 = _
  var stsMock : AWSSecurityTokenService = _
  var sparkConfMock : SparkConf = _

  val s3Bucket = "a"
  val s3Prefix = "b"
  val s3TrainingPrefix = "b/test-training-job"
  val s3DataPrefix = "b/test-training-job/data.pbr"

  before {
    dataset = mock[Dataset[String]]
    dataUploaderMock = mock[DataUploader]
    when(dataUploaderMock.uploadData(any[S3DataPath], any[Dataset[_]]))
      .thenReturn(ObjectPrefixUploadResult(S3DataPath(s3Bucket, s3TrainingPrefix)))
    timeProviderMock = mock[TimeProvider]

    sagemakerMock = mock[AmazonSageMaker]
    s3Mock = mock[AmazonS3]
    stsMock = mock[AWSSecurityTokenService]
    sparkConfMock = mock[SparkConf]

    val sparkSessionMock = mock[SparkSession]
    val sparkContextMock = mock[SparkContext]

    when(dataset.sparkSession).thenReturn(sparkSessionMock)
    when(sparkSessionMock.sparkContext).thenReturn(sparkContextMock)
    when(sparkContextMock.getConf).thenReturn(sparkConfMock)

    val objectSummaryMock = mock[S3ObjectSummary]
    when(objectSummaryMock.getKey).thenReturn(s3DataPrefix)
    val objectListMock = mock[ObjectListing]
    when(objectListMock.getObjectSummaries).thenReturn(util.Arrays.asList(objectSummaryMock))
    when(s3Mock.listObjects(s3Bucket, s3TrainingPrefix)).thenReturn(objectListMock)
    when(s3Mock.getRegionName).thenReturn("region")
  }

  "SageMakerEstimator" should "generate a UID" in {

    val estimator = new DummyEstimator()
    val estimator2 = new DummyEstimator(uid = "blah")

    assert(estimator.toString().startsWith("sagemaker"))
    assert(estimator.uid != estimator2.uid)
  }

  it should "have empty hyperparameter map when no params defined" in {
    val estimator = new DummyEstimator()
    assert(estimator.makeHyperParameters() isEmpty)
  }

  it should "have correct hyperparameter when empty params defined" in {
    val estimator = new DummyEstimator() {
      val stringParam: Param[String] = new Param(this, "stringParam", "")
      val intParam: IntParam = new IntParam(this, "intParam", "")
      val booleanParam: BooleanParam = new BooleanParam(this, "booleanParam", "")
      val otherStringParam: Param[String] = new Param[String](this, "otherStringParam", "")
    }
    assert(estimator.makeHyperParameters() == collection.immutable.Map().asJava)
  }

  it should "record the latest training job after calling fit()" in {
    val estimator = new DummyEstimator()
    assert (estimator.latestTrainingJob.isEmpty)

    when(timeProviderMock.currentTimeMillis).thenReturn(0)
    setupCreateTrainingJobResult()
    setupDescribeTrainingJobResponses(TrainingJobStatus.Completed)
    val model = estimator.fit(dataset)
    assert (estimator.latestTrainingJob.nonEmpty)
  }

  it should "select only the projected columns of the dataset to a given s3 location in fit()" in {
    val estimator = new DummyEstimator()
    when(timeProviderMock.currentTimeMillis).thenReturn(0)
    setupCreateTrainingJobResult()
    setupDescribeTrainingJobResponses(TrainingJobStatus.Completed)
    val model = estimator.fit(dataset)
    verify(dataset).select(s3Bucket, s3Prefix)
  }

  it should "attempt all the columns of the dataset to a given s3 location in fit()" in {
    val estimator = new DummyEstimator(dummyTrainingProjectedColumns = None)
    when(timeProviderMock.currentTimeMillis).thenReturn(0)
    setupCreateTrainingJobResult()
    setupDescribeTrainingJobResponses(TrainingJobStatus.Completed)
    val model = estimator.fit(dataset)
    verify(dataset, times(0)).select(any[String], any[String])
    verify(dataUploaderMock).uploadData(estimator.dummyS3InputDataPathWithTrainingJobName
      .asInstanceOf[S3DataPath], dataset)
  }

  it should "attempt all the columns of the dataset to a given s3 location in fit() if given an " +
    "empty list of columns names" in {
    val estimator = new DummyEstimator(dummyTrainingProjectedColumns = Some(List[String]()))
    when(timeProviderMock.currentTimeMillis).thenReturn(0)
    setupCreateTrainingJobResult()
    setupDescribeTrainingJobResponses(TrainingJobStatus.Completed)
    val model = estimator.fit(dataset)
    verify(dataset, times(0)).select(any[String], any[String])
    verify(dataUploaderMock).uploadData(estimator.dummyS3InputDataPathWithTrainingJobName
      .asInstanceOf[S3DataPath], dataset)
  }

  it should "correctly create a training job request from training properties" in {
    val estimator = new DummyEstimator()
    when(timeProviderMock.currentTimeMillis).thenReturn(0)
    setupCreateTrainingJobResult()
    setupDescribeTrainingJobResponses(TrainingJobStatus.Completed)

    when(dataUploaderMock.uploadData(any[S3DataPath], any[Dataset[_]]))
      .thenReturn(ObjectPrefixUploadResult(
        estimator.trainingInputS3DataPath.asInstanceOf[S3DataPath]))
    estimator.fit(dataset)

    val createTrainingJobRequestCaptor = ArgumentCaptor.forClass(classOf[CreateTrainingJobRequest])
    verify(sagemakerMock).createTrainingJob(createTrainingJobRequestCaptor.capture())

    val createTrainingJobArgument = createTrainingJobRequestCaptor.getValue
    assert(estimator.trainingImage == createTrainingJobArgument.getAlgorithmSpecification
      .getTrainingImage)

    assert(estimator.trainingInputMode == createTrainingJobArgument.getAlgorithmSpecification
      .getTrainingInputMode)
    assert(estimator.trainingCompressionCodec.get == createTrainingJobArgument.getInputDataConfig
      .get(0).getCompressionType)
    assert(estimator.trainingChannelName == createTrainingJobArgument.getInputDataConfig.get(0)
      .getChannelName)
    assert(estimator.trainingContentType.get == createTrainingJobArgument.getInputDataConfig.get(0)
      .getContentType)
    assert(estimator.trainingInputS3DataPath.asInstanceOf[S3DataPath].toS3UriString
      == createTrainingJobArgument.getInputDataConfig.get(0).getDataSource.getS3DataSource.getS3Uri)
    assert(estimator.trainingS3DataDistribution == createTrainingJobArgument.getInputDataConfig
      .get(0).getDataSource.getS3DataSource.getS3DataDistributionType)
    assert(estimator.dummyS3OutputDataPathWithTrainingJobName.toS3UriString
      == createTrainingJobArgument.getOutputDataConfig.getS3OutputPath)
    assert(estimator.trainingInstanceCount == createTrainingJobArgument
      .getResourceConfig.getInstanceCount)
    assert(estimator.trainingInstanceType == createTrainingJobArgument.getResourceConfig
      .getInstanceType)
    assert(estimator.sagemakerRole
      .asInstanceOf[IAMRole].role == createTrainingJobArgument.getRoleArn)
    assert(estimator.trainingMaxRuntimeInSeconds == createTrainingJobArgument.getStoppingCondition
      .getMaxRuntimeInSeconds)
    assert(estimator.trainingKmsKeyId.get == createTrainingJobArgument.getOutputDataConfig
      .getKmsKeyId)
  }

  it should "have correct hyperparameter map when default and non default params set" in {

    val estimator = new DummyEstimator() {
      val stringParam : Param[String] = new Param(this, "stringParam", "")
      val intParam : IntParam = new IntParam(this, "intParam", "")
      val booleanParam : BooleanParam = new BooleanParam(this, "booleanParam", "")
      val otherStringParam : Param[String] = new Param[String](this, "otherStringParam", "")
      setDefault(stringParam, "default")
      setDefault(intParam, 55)
    }
    estimator.set(estimator.intParam, 66)
    estimator.set(estimator.otherStringParam, "Elizabeth")
    assert(estimator.makeHyperParameters() == Map(
      "stringParam" -> "default",
      "intParam" -> "66",
      "otherStringParam" -> "Elizabeth").asJava)
  }

  it should "poll for training job completion" in {
    val estimator = new DummyEstimator()
    when(timeProviderMock.currentTimeMillis).thenReturn(0)
    setupCreateTrainingJobResult()
    setupDescribeTrainingJobResponses(TrainingJobStatus.InProgress, TrainingJobStatus.InProgress,
      TrainingJobStatus.Completed)
    val sagemakerModel = estimator.fit(dataset)
    verify(sagemakerMock, times(4)).describeTrainingJob(any[DescribeTrainingJobRequest])
    verify(timeProviderMock, times(2)).sleep(SageMakerEstimator.TrainingJobPollInterval.toMillis)
    assert(sagemakerModel.uid == estimator.uid)
  }

  it should "ignore transient failures when polling for training completion" in {
    val estimator = new DummyEstimator()
    when(timeProviderMock.currentTimeMillis).thenReturn(0)
    setupCreateTrainingJobResult()
    val retryableException = new AmazonSageMakerException("transient failure")
    retryableException.setStatusCode(500)
    when(sagemakerMock.describeTrainingJob(any[DescribeTrainingJobRequest]))
      .thenThrow(retryableException)
      .thenReturn(statusToResult(TrainingJobStatus.InProgress))
      .thenReturn(statusToResult(TrainingJobStatus.Completed))
    estimator.fit(dataset)
    verify(sagemakerMock, times(4)).describeTrainingJob(any[DescribeTrainingJobRequest])
    verify(timeProviderMock, times(2)).sleep(SageMakerEstimator.TrainingJobPollInterval.toMillis)
  }

  it should "throw an exception if the training job fails to create" in {
    when(sagemakerMock.createTrainingJob(any[CreateTrainingJobRequest]))
      .thenThrow(new AmazonSageMakerException("EASE is down."))
    val estimator = new DummyEstimator()

    when(timeProviderMock.currentTimeMillis).thenReturn(0)
    val caught = intercept[RuntimeException] {
      estimator.fit(dataset)
    }

    verify(sagemakerMock, never()).describeTrainingJob(any[DescribeTrainingJobRequest])
  }

  it should "throw an exception when training fails" in {
    val estimator = new DummyEstimator()
    when(timeProviderMock.currentTimeMillis).thenReturn(0)
    setupDescribeTrainingJobResponses(TrainingJobStatus.InProgress, TrainingJobStatus.InProgress,
      TrainingJobStatus.Failed)
    val caught = intercept[RuntimeException] {
      estimator.fit(dataset)
    }
  }

  it should "throw an exception when polling exceeds training timeout" in {
    val estimator = new DummyEstimator()

    when(timeProviderMock.currentTimeMillis)
      .thenReturn(0) // starting s3 upload time
      .thenReturn(0) // starting training time
      .thenReturn(1) // await training completion start time
      .thenReturn(2) // first while loop check
      .thenReturn(2) // second while loop check
      // third while loop check - should exit loop
      .thenReturn(estimator.trainingJobTimeout.toMillis + 1)

    setupCreateTrainingJobResult()
    setupDescribeTrainingJobResponses(TrainingJobStatus.InProgress)
    val caught = intercept[RuntimeException] {
      estimator.fit(dataset)
    }

    verify(sagemakerMock, times(2)).describeTrainingJob(any[DescribeTrainingJobRequest])
    verify(timeProviderMock, times(2)).sleep(SageMakerEstimator.TrainingJobPollInterval.toMillis)
  }

  it should "create a training job request with a manifest file if running on EMRFS" in {
    val estimator = new DummyEstimator()
    val trainingJobRequest = estimator.buildCreateTrainingJobRequest("blah",
      ManifestDataUploadResult(new S3DataPath("bucket", "objectPath")), sparkConfMock)
    val s3DataType = trainingJobRequest.getInputDataConfig.get(0).getDataSource.getS3DataSource
      .getS3DataType
    assert(S3DataType.ManifestFile.toString == s3DataType)
  }

  it should "create a training job request without a manifest file for input datasource if not " +
    "running on EMRFS" in {
    val estimator = new DummyEstimator()
    val trainingJobRequest = estimator.buildCreateTrainingJobRequest("blah",
      ObjectPrefixUploadResult(new S3DataPath("bucket", "objectPath")), sparkConfMock)
    val s3DataType = trainingJobRequest.getInputDataConfig.get(0).getDataSource.getS3DataSource
      .getS3DataType
    assert(S3DataType.S3Prefix.toString == s3DataType)
  }

  it should "resolve s3 locations from configuration" in {
    val estimator = new DummyEstimator()
    val trainingJobName = "training"
    when(sparkConfMock.get("test-config-key")).thenReturn ("s3://bucket/path")
    val dp = estimator.resolveS3Path(S3PathFromConfig("test-config-key"), trainingJobName,
      sparkConfMock)
    assert(S3DataPath("bucket", "path/training") == dp)
    assert(dp.objectPath.endsWith(trainingJobName))
  }

  it should "resolve s3 locations with random paths from configuration" in {
    val estimator = new DummyEstimator()
    val trainingJobName = "training"
    when(sparkConfMock.get("test-config-key")).thenReturn ("bucket")
    val dp = estimator.resolveS3Path(S3PathFromConfig("test-config-key"), trainingJobName,
      sparkConfMock)
    assert (dp.bucket == "bucket")
    assert (dp.objectPath.length == 45)
    assert (dp.objectPath.endsWith(trainingJobName))
  }

  it should "resolve role arn from configuration" in {
    val estimator = new DummyEstimator()
    when(sparkConfMock.get("test-config-key")).thenReturn("arn-role")
    assert(IAMRole("arn-role") == estimator.resolveRoleARN(IAMRoleFromConfig("test-config-key"),
      sparkConfMock))
  }

  it should "resolve role arn from role" in {
    val estimator = new DummyEstimator()
    assert(IAMRole("arn-role") == estimator.resolveRoleARN(IAMRole("arn-role"), sparkConfMock))
  }

  it should "create bucket and prefix with training job name" in {
    val estimator = new DummyEstimator()
    val mockAccount = "1234"
    val mockResult = new GetCallerIdentityResult().withAccount(mockAccount)
    val trainingJobName = "training"
    when(stsMock.getCallerIdentity(any[GetCallerIdentityRequest])).thenReturn(mockResult)
    val dp = estimator.resolveS3Path(S3AutoCreatePath(), trainingJobName, sparkConfMock)
    verify(s3Mock, times(1)).createBucket("1234-sagemaker-region")
    assert("1234-sagemaker-region" == dp.bucket)
    assert(dp.objectPath.endsWith(trainingJobName))
  }

  it should "refuse to create a bucket if it already exists" in {
    when(timeProviderMock.currentTimeMillis).thenReturn(0)
    val mockAccount = "1234"
    val mockResult = new GetCallerIdentityResult().withAccount(mockAccount)
    when(stsMock.getCallerIdentity(any[GetCallerIdentityRequest])).thenReturn(mockResult)

    when(s3Mock.createBucket(any[String])).thenThrow(new AmazonS3Exception(
      "not a bucket exists exception"))
    val estimator = new DummyEstimator(dummyTrainingInputS3DataPath = S3AutoCreatePath())
    intercept[AmazonS3Exception] {
      estimator.fit(dataset)
    }
  }

  it should "fail to create bucket on exception" in {
    when(timeProviderMock.currentTimeMillis).thenReturn(0)
    val mockAccount = "1234"
    val mockResult = new GetCallerIdentityResult().withAccount(mockAccount)
    when(stsMock.getCallerIdentity(any[GetCallerIdentityRequest])).thenReturn(mockResult)

    when(s3Mock.createBucket(any[String])).thenThrow(new AmazonS3Exception(
      "not a bucket exists exception"))
    val estimator = new DummyEstimator(dummyTrainingInputS3DataPath = S3AutoCreatePath())
    intercept[AmazonS3Exception] {
      estimator.fit(dataset)
    }
  }

  it should "take hyperparameter map in constructor" in {
    val estimator = new DummyEstimator(dummyHyperParameters = Map(
      s3Bucket -> "a value", s3Prefix -> "55"))
    estimator.makeHyperParameters().equals(Map(s3Bucket -> "a value", s3Prefix -> "55").asJava)
  }

  it should "remove the training data when training completed" in {
    when(timeProviderMock.currentTimeMillis).thenReturn(0)
    val mockAccount = "1234"
    val mockResult = new GetCallerIdentityResult().withAccount(mockAccount)
    when(stsMock.getCallerIdentity(any[GetCallerIdentityRequest])).thenReturn(mockResult)

    val estimator = new DummyEstimator()

    when(sagemakerMock.describeTrainingJob(any[DescribeTrainingJobRequest]))
      .thenReturn(statusToResult(TrainingJobStatus.InProgress))
      .thenReturn(statusToResult(TrainingJobStatus.Completed))

    estimator.fit(dataset)

    verify(s3Mock).deleteObject(s3Bucket, s3DataPrefix)
    verify(s3Mock).deleteObject(s3Bucket, s3TrainingPrefix)
  }

  it should "remove the training data when training failed" in {
    when(timeProviderMock.currentTimeMillis).thenReturn(0)
    val mockAccount = "1234"
    val mockResult = new GetCallerIdentityResult().withAccount(mockAccount)
    when(stsMock.getCallerIdentity(any[GetCallerIdentityRequest])).thenReturn(mockResult)

    val estimator = new DummyEstimator()

    intercept[RuntimeException] {
      estimator.fit(dataset)
    }

    verify(s3Mock).deleteObject(s3Bucket, s3DataPrefix)
    verify(s3Mock).deleteObject(s3Bucket, s3TrainingPrefix)
  }

  it should "swallow the s3 exception if failed to remove training data" in {
    when(s3Mock.deleteObject(any[String], any[String])).thenThrow(new SdkClientException(
      "failed to delete training data"))
    when(timeProviderMock.currentTimeMillis).thenReturn(0)
    val mockAccount = "1234"
    val mockResult = new GetCallerIdentityResult().withAccount(mockAccount)
    when(stsMock.getCallerIdentity(any[GetCallerIdentityRequest])).thenReturn(mockResult)

    val estimator = new DummyEstimator()

    when(sagemakerMock.describeTrainingJob(any[DescribeTrainingJobRequest]))
      .thenReturn(statusToResult(TrainingJobStatus.InProgress))
      .thenReturn(statusToResult(TrainingJobStatus.Completed))

    estimator.fit(dataset)

    verify(s3Mock).deleteObject(any[String], any[String])
  }

  it should "keep the training data when DeleteAfterTrain is false" in {
    when(timeProviderMock.currentTimeMillis).thenReturn(0)
    val mockAccount = "1234"
    val mockResult = new GetCallerIdentityResult().withAccount(mockAccount)
    when(stsMock.getCallerIdentity(any[GetCallerIdentityRequest])).thenReturn(mockResult)

    val estimator = new DummyEstimator(dummyDeleteAfterTraining = false)

    when(sagemakerMock.describeTrainingJob(any[DescribeTrainingJobRequest]))
      .thenReturn(statusToResult(TrainingJobStatus.InProgress))
      .thenReturn(statusToResult(TrainingJobStatus.Completed))

    estimator.fit(dataset)

    verify(s3Mock, never).deleteObject(any[String], any[String])
  }

  val dummyModelArtifactLocation : String = "s3://bucket/string"

  case class DummyNamePolicy(val prefix : String = "") extends NamePolicy {
    val uid = "test"

    val trainingJobName = uid + "-training-job"
    val modelName = uid + "-model"
    val endpointConfigName = uid + "-endpoint-config"
    val endpointName = uid + "-endpoint"
  }

  class DummyEstimator (
    override val uid : String = "sagemaker",
    val dummyTrainingImage : String = "training-image",
    val dummyModelImage : String = "model-image",
    val dummyRequestRowSerializer: RequestRowSerializer = new LibSVMRequestRowSerializer(),
    val dummyResponseRowDeserializer: ResponseRowDeserializer =
      new LibSVMResponseRowDeserializer(10),
    val dummySageMakerRole : IAMRoleResource = IAMRole("dummy-role"),
    val dummyTrainingInputS3DataPath : S3Resource = S3DataPath(s3Bucket, s3Prefix),
    val dummyTrainingOutputS3DataPath : S3Resource = S3DataPath(s3Bucket, s3Prefix),
    val dummyTrainingInstanceType : String = "m4.large",
    val dummyTrainingInstanceCount : Int = 1,
    val dummyTrainingInstanceVolumeSizeInGB : Int = 1024,
    val dummyTrainingProjectedColumns : Option[List[String]] = Some(List(s3Bucket, s3Prefix)),
    val dummyTrainingChannelName : String = "training",
    val dummyTrainingContentType : Option[String] = Some("application/x-record-protobuf"),
    val dummyTrainingS3DataDistribution : String = S3DataDistribution.ShardedByS3Key.toString,
    val dummyTrainingSparkDataFormat : String = "sagemaker",
    val dummyTrainingSparkDataFormatOptions : collection.immutable.Map[String, String] =
      collection.immutable.Map(),
    val dummyTrainingInputMode : String = TrainingInputMode.File.toString,
    val dummyTrainingCompressionCodec : Option[String] = Some("codec"),
    val dummytrainingMaxRuntimeInSeconds : Int = 24 * 60 * 60,
    val dummyTrainingKmsKeyId : Option[String] = Some("kms"),
    val dummyModelEnvironmentVariables : collection.immutable.Map[String, String] =
      collection.immutable.Map(),
    val dummyEndpointInstanceType : String = "m4.large",
    val dummyendpointInitialInstanceCount : Int = 1,
    val dummyEndpointCreationPolicy : EndpointCreationPolicy =
      EndpointCreationPolicy.CREATE_ON_TRANSFORM,
    val dummyModelPrependInputRowsToTransformationRows : Boolean = true,
    val dummyDeleteAfterTraining : Boolean = true,
    val dummyNamePolicy : NamePolicy = DummyNamePolicy(),
    val dummyHyperParameters : Map[String, String] = Map()) extends SageMakerEstimator (
    dummyTrainingImage,
    dummyModelImage,
    dummySageMakerRole,
    dummyTrainingInstanceType,
    dummyTrainingInstanceCount,
    dummyEndpointInstanceType,
    dummyendpointInitialInstanceCount,
    dummyRequestRowSerializer,
    dummyResponseRowDeserializer,
    dummyTrainingInputS3DataPath,
    dummyTrainingOutputS3DataPath,
    dummyTrainingInstanceVolumeSizeInGB,
    dummyTrainingProjectedColumns,
    dummyTrainingChannelName,
    dummyTrainingContentType,
    dummyTrainingS3DataDistribution,
    dummyTrainingSparkDataFormat,
    dummyTrainingSparkDataFormatOptions,
    dummyTrainingInputMode,
    dummyTrainingCompressionCodec,
    dummytrainingMaxRuntimeInSeconds,
    dummyTrainingKmsKeyId,
    dummyModelEnvironmentVariables,
    dummyEndpointCreationPolicy,
    sagemakerMock,
    s3Mock,
    stsMock,
    dummyModelPrependInputRowsToTransformationRows,
    dummyDeleteAfterTraining,
    new NamePolicyFactory { override def createNamePolicy: NamePolicy = DummyNamePolicy() },
    uid,
    dummyHyperParameters) {
    this.timeProvider = timeProviderMock
    this.dataUploader = dataUploaderMock

    def dummyS3InputDataPathWithTrainingJobName : S3DataPath =
      resolveS3Path(dummyTrainingInputS3DataPath, dummyNamePolicy.trainingJobName, sparkConfMock)
    def dummyS3OutputDataPathWithTrainingJobName : S3DataPath =
      resolveS3Path(dummyTrainingOutputS3DataPath, dummyNamePolicy.trainingJobName, sparkConfMock)
  }

  private def setupCreateTrainingJobResult() = {
    val fakeArn = "arn"
    val mockCreateTrainingJobResult = new CreateTrainingJobResult().withTrainingJobArn(fakeArn)
    when(sagemakerMock.createTrainingJob(any[CreateTrainingJobRequest])).thenReturn(
      mockCreateTrainingJobResult)
  }

  private def statusToResult(status: TrainingJobStatus): DescribeTrainingJobResult = {
    val modelArtifacts = new ModelArtifacts().withS3ModelArtifacts(dummyModelArtifactLocation)
    new DescribeTrainingJobResult().withTrainingJobStatus(status).withModelArtifacts(modelArtifacts)
  }

  private def setupDescribeTrainingJobResponses(firstStatus: TrainingJobStatus,
                                                moreStatuses : TrainingJobStatus*) = {
    when(sagemakerMock.describeTrainingJob(any[DescribeTrainingJobRequest])).
      thenReturn(statusToResult(firstStatus),
      moreStatuses.map(statusToResult): _*)
  }

}

