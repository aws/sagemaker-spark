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

import java.time.Duration
import java.util.UUID

import scala.collection.immutable.Map
import scala.jdk.CollectionConverters._

import com.amazonaws.SdkBaseException
import com.amazonaws.retry.RetryUtils
import com.amazonaws.services.s3.{AmazonS3, AmazonS3ClientBuilder}
import com.amazonaws.services.s3.model.AmazonS3Exception
import com.amazonaws.services.securitytoken.{AWSSecurityTokenService, AWSSecurityTokenServiceClientBuilder}
import com.amazonaws.services.securitytoken.model.GetCallerIdentityRequest

import org.apache.spark.SparkConf
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.sql.types.StructType

import com.amazonaws.services.sagemaker.{AmazonSageMaker, AmazonSageMakerClientBuilder}
import com.amazonaws.services.sagemaker.model._
import com.amazonaws.services.sagemaker.sparksdk.EndpointCreationPolicy.EndpointCreationPolicy
import com.amazonaws.services.sagemaker.sparksdk.internal.{DataUploader, DataUploadResult, InternalUtils, ManifestDataUploadResult, ObjectPrefixUploadResult, SystemTimeProvider, TimeProvider}
import com.amazonaws.services.sagemaker.sparksdk.transformation.{RequestRowSerializer, ResponseRowDeserializer}

/**
  * Determines whether and when to create the Endpoint and other Hosting resources.
  *
  * CREATE_ON_CONSTRUCT - create the Endpoint upon creation of the SageMakerModel, at the end of
  * fit()
  * CREATE_ON_TRANSFORM - create the Endpoint upon invocation of SageMakerModel.transform()
  * DO_NOT_CREATE - do not create the Endpoint
  */
object EndpointCreationPolicy extends Enumeration {
  type EndpointCreationPolicy = Value
  val CREATE_ON_CONSTRUCT, CREATE_ON_TRANSFORM, DO_NOT_CREATE = Value
}

object SageMakerEstimator {
  var TrainingJobPollInterval = Duration.ofSeconds(5)
}

/**
  * Adapts a SageMaker learning Algorithm to a Spark Estimator. Fits a [[SageMakerModel]] by
  * running a SageMaker Training Job on a Spark Dataset. Each call to
  * [[SageMakerEstimator!.fit* fit]] submits a new SageMaker Training Job, creates a new SageMaker
  * Model, and creates a new SageMaker Endpoint Config. A new Endpoint is either created by or
  * the returned SageMakerModel is configured to generate an Endpoint on SageMakerModel transform.
  *
  * On fit, the input [[org.apache.spark.sql.Dataset dataset]] is serialized with the specified
  * [[trainingSparkDataFormat]] using the specified [[trainingSparkDataFormatOptions]] and uploaded
  * to an S3 location specified by [[trainingInputS3DataPath]]. The serialized Dataset
  * is compressed with [[trainingCompressionCodec]], if not None.
  *
  * [[trainingProjectedColumns]] can be used to control which columns on the input Dataset are
  * transmitted to SageMaker. If not None, then only those column names will be serialized as input
  * to the SageMaker Training Job.
  *
  * A Training Job is created with the uploaded Dataset being input to the specified
  * [[trainingChannelName]], with the specified [[trainingInputMode]]. The algorithm is specified
  * [[trainingImage]], a Docker image URI reference.
  * The Training Job is created with [[trainingInstanceCount]] instances of type
  * [[trainingInstanceType]]. The Training Job will time-out after [[trainingMaxRuntimeInSeconds]],
  * if not None.
  *
  * SageMaker Training Job hyperparameters are built from the [[org.apache.spark.ml.param.Param]]s
  * set on this Estimator. Param objects set on this Estimator are retrieved during fit and
  * converted to a SageMaker Training Job hyperparameter Map.
  * Param objects are iterated over by invoking [[org.apache.spark.ml.param.Params.params params]]
  * on this Estimator.
  * Param objects with neither a default value nor a set value are ignored. If a Param is not set
  * but has a default value, the default value will be used. Param values are converted to SageMaker
  * hyperparameter String values by invoking [[org.apache.spark.ml.param.Params.toString toString]]
  * on the Param value.
  *
  * SageMaker uses the IAM Role with ARN [[sagemakerRole]] to access the input and output S3
  * buckets and trainingImage if the image is hosted in ECR. SageMaker Training Job output is
  * stored in a Training Job specific sub-prefix of [[trainingOutputS3DataPath]]. This contains
  * the SageMaker Training Job output file as well as the SageMaker Training Job model file.
  *
  * After the Training Job is created, this Estimator will poll for success. Upon success an
  * [[SageMakerModel]] is created and returned from fit. The SageMakerModel is created with a
  * [[modelImage]] Docker image URI, defining the SageMaker model primary container and with
  * [[modelEnvironmentVariables]] environment variables.
  * Each SageMakerModel has a corresponding SageMaker hosting Endpoint. This Endpoint runs on at
  * least [[endpointInitialInstanceCount]] instances of type [[endpointInstanceType]]. The
  * Endpoint is created either during construction of the SageMakerModel or on the first call to
  * [[SageMakerModel!.transform* transform]], controlled by [[endpointCreationPolicy]]. Each
  * Endpoint instance runs with [[sagemakerRole]] IAMRole.
  *
  * The [[SageMakerModel!.transform* transform]] method on SageMakerModel uses
  * [[requestRowSerializer]] to serialize Rows from the Dataset undergoing transformation, to
  * requests on the hosted SageMaker Endpoint. The [[responseRowDeserializer]] is used to convert
  * the response from the Endpoint to a series of Rows, forming the transformed Dataset. If
  * [[modelPrependInputRowsToTransformationRows]] is true, then each transformed Row is also
  * prepended with its corresponding input Row.
  *
  * @param trainingImage A SageMaker Training Job Algorithm Specification Training Image Docker
  *                      image URI.
  * @param modelImage A SageMaker Model hosting Docker image URI.
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
  * @param hyperParameters A map from hyperParameter names to their respective values for training.
  * @param trainingInputS3DataPath An S3 location to upload SageMaker Training Job input data to.
  * @param trainingOutputS3DataPath An S3 location for SageMaker to store Training Job output data
  *                                 to.
  * @param trainingInstanceVolumeSizeInGB The EBS volume size in gigabytes of each instance
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
class SageMakerEstimator(val trainingImage: String,
                         val modelImage: String,
                         val sagemakerRole: IAMRoleResource = IAMRoleFromConfig(),
                         val trainingInstanceType: String,
                         val trainingInstanceCount: Int,
                         val endpointInstanceType: String,
                         val endpointInitialInstanceCount: Int,
                         val requestRowSerializer: RequestRowSerializer,
                         val responseRowDeserializer: ResponseRowDeserializer,
                         val trainingInputS3DataPath: S3Resource = S3AutoCreatePath(),
                         val trainingOutputS3DataPath: S3Resource = S3AutoCreatePath(),
                         val trainingInstanceVolumeSizeInGB: Int = 1024,
                         val trainingProjectedColumns: Option[List[String]] = None,
                         val trainingChannelName: String = "train",
                         val trainingContentType: Option[String] = None,
                         val trainingS3DataDistribution: String
                           = S3DataDistribution.ShardedByS3Key.toString,
                         val trainingSparkDataFormat: String = "sagemaker",
                         val trainingSparkDataFormatOptions: Map[String, String] = Map(),
                         val trainingInputMode: String = TrainingInputMode.File.toString,
                         val trainingCompressionCodec: Option[String] = None,
                         val trainingMaxRuntimeInSeconds: Int = 24 * 60 * 60,
                         val trainingKmsKeyId: Option[String] = None,
                         val modelEnvironmentVariables: Map[String, String] = Map(),
                         val endpointCreationPolicy: EndpointCreationPolicy
                           = EndpointCreationPolicy.CREATE_ON_CONSTRUCT,
                         val sagemakerClient : AmazonSageMaker
                           = AmazonSageMakerClientBuilder.defaultClient,
                         val s3Client: AmazonS3 = AmazonS3ClientBuilder.defaultClient(),
                         val stsClient: AWSSecurityTokenService
                           = AWSSecurityTokenServiceClientBuilder.defaultClient(),
                         val modelPrependInputRowsToTransformationRows: Boolean = true,
                         val deleteStagingDataAfterTraining: Boolean = true,
                         val namePolicyFactory: NamePolicyFactory = new RandomNamePolicyFactory(),
                         override val uid: String = Identifiable.randomUID("sagemaker"),
                         val hyperParameters: Map[String, String] = Map())
  extends Estimator[SageMakerModel] {

  private[sparksdk] var timeProvider: TimeProvider = new SystemTimeProvider
  private[sparksdk] var dataUploader: DataUploader = new DataUploader(trainingSparkDataFormat,
    trainingSparkDataFormatOptions)

  private[sparksdk] val trainingJobTimeout = Duration.ofHours(trainingMaxRuntimeInSeconds)

  private[sparksdk] var latestTrainingJob: Option[String] = None

  /**
    * Builds a SageMaker Training Job hyper-parameter map from the Params set on this
    * defined both on the input set and on the
    *
    * @return a SageMaker hyper-parameter map
    */
  private[sparksdk] def makeHyperParameters() : java.util.Map[String, String] = {
    val trainingJobHyperParameters : scala.collection.mutable.Map[String, String] =
      scala.collection.mutable.Map() ++ hyperParameters

    params.filter(p => hasDefault(p) || isSet(p)) map {
      case p => (p.name, this.getOrDefault(p).toString)
    } foreach {
      case (key, value) => trainingJobHyperParameters.put(key, value)
    }
    trainingJobHyperParameters.asJava
  }

  private[sparksdk] def resolveS3Path(s3Resource : S3Resource,
                                      trainingJobName : String, config : SparkConf): S3DataPath = {
    s3Resource match {
      case s3DataPath : S3DataPath =>
        new S3DataPath(s3DataPath.bucket, s3DataPath.objectPath + "/" + trainingJobName)
      case S3PathFromConfig(configKey) =>
        val configValue = config.get(configKey)
        if(configValue.matches("^s3[a|n]?://.+")) {
          val s3URI = configValue.stripSuffix("/") + "/" + trainingJobName
          S3DataPath.fromS3URI(s3URI)
        } else {
          val prefix = UUID.randomUUID().toString + "/" + trainingJobName
          S3DataPath(configValue, prefix)
        }
      case S3AutoCreatePath() =>
        val account = stsClient.getCallerIdentity(new GetCallerIdentityRequest).getAccount
        val region = s3Client.getRegionName
        val bucketName = s"$account-sagemaker-$region"
        try {
          s3Client.createBucket(bucketName)
          log.info(s"Created bucket $bucketName.")
        } catch {
          case ex : AmazonS3Exception =>
            // This exception is thrown if the S3 client is in us-east-1 but the bucket is not.
            if (Option(ex.getErrorCode).getOrElse("").contains("BucketAlreadyOwnedByYou")) {
              log.info(s"Using bucket $bucketName, which you already own.")
            } else if (Option(ex.getErrorCode).getOrElse("")
              .contains("AuthorizationHeaderMalformed")) {
              log.info(s"Bucket $bucketName already exists in a different region, " +
                s"not ${s3Client.getRegionName}. Attempting to use bucket $bucketName")
            } else {
              throw ex
            }
        }
        val prefix = UUID.randomUUID().toString + "/" + trainingJobName
        S3DataPath(bucketName, prefix)
    }
  }

  private[sparksdk] def resolveRoleARN(iAMRoleResource: IAMRoleResource,
                                       config : SparkConf) : IAMRole = {
    iAMRoleResource match {
      case iamRole : IAMRole => iamRole
      case IAMRoleFromConfig(configKey) => IAMRole(config.get(configKey))
    }
  }

  /**
    * Fits a [[SageMakerModel]] on dataSet by running a SageMaker training job.
    */
  override def fit(dataSet: Dataset[_]): SageMakerModel = {
    transformSchema(dataSet.schema, logging = true)
    val namePolicy = namePolicyFactory.createNamePolicy
    val trainingJobName = namePolicy.trainingJobName

    val conf = dataSet.sparkSession.sparkContext.getConf
    val inputPath = resolveS3Path(trainingInputS3DataPath, trainingJobName, conf)

    val startingS3UploadTime = this.timeProvider.currentTimeMillis

    val dataUploadResults = trainingProjectedColumns match {
      case Some(columns) if !columns.isEmpty => dataUploader.uploadData(inputPath,
        dataSet.select(columns.head, columns.tail: _*))
      case _ => dataUploader.uploadData(inputPath, dataSet)
    }

    val s3UploadTime = this.timeProvider.getElapsedTimeInSeconds(startingS3UploadTime)
    log.info(s"S3 Upload Time: $s3UploadTime s")

    try {
      log.info(s"Creating training job with name $trainingJobName")
      latestTrainingJob = Some(trainingJobName)
      val createTrainingJobRequest = buildCreateTrainingJobRequest(trainingJobName,
        dataUploadResults, conf)

      log.info(s"CreateTrainingJobRequest: ${createTrainingJobRequest.toString}")
      runTrainingJob(createTrainingJobRequest, trainingJobName)
    } finally {
      if (deleteStagingDataAfterTraining) {
        log.info(s"Deleting training data ${inputPath.toS3UriString} of job with" +
          s" name $trainingJobName")
        deleteTrainingData(inputPath)
      }
    }

    val describeTrainingJobRequest = new DescribeTrainingJobRequest()
      .withTrainingJobName(trainingJobName)
    InternalUtils.applyUserAgent(describeTrainingJobRequest)
    val modelS3URI = sagemakerClient.describeTrainingJob(describeTrainingJobRequest)
        .getModelArtifacts
        .getS3ModelArtifacts

    log.info(s"Model S3 URI: $modelS3URI")
    new SageMakerModel(
      Some(endpointInstanceType),
      Some(endpointInitialInstanceCount),
      requestRowSerializer,
      responseRowDeserializer,
      Option.empty,
      Some(modelImage),
      Some(S3DataPath.fromS3URI(modelS3URI)),
      modelEnvironmentVariables,
      Some(resolveRoleARN(sagemakerRole, conf).role),
      endpointCreationPolicy,
      sagemakerClient,
      modelPrependInputRowsToTransformationRows,
      namePolicy,
      uid)
  }

  private[sparksdk] def buildCreateTrainingJobRequest(trainingJobName: String,
                                                      dataUploadResults: DataUploadResult,
                                                      conf: SparkConf): CreateTrainingJobRequest = {
    val createTrainingJobRequest = new CreateTrainingJobRequest()
    InternalUtils.applyUserAgent(createTrainingJobRequest)

    createTrainingJobRequest.withTrainingJobName(trainingJobName)

    val algorithmSpecification = new AlgorithmSpecification()
    algorithmSpecification.setTrainingImage(trainingImage)
    algorithmSpecification.setTrainingInputMode(trainingInputMode)
    createTrainingJobRequest.setAlgorithmSpecification(algorithmSpecification)

    var hyperParameters = makeHyperParameters()
    if (hyperParameters.isEmpty) {
      hyperParameters = null
    }
    createTrainingJobRequest.withHyperParameters(hyperParameters)
    val inputS3Path = dataUploadResults.s3DataPath

    val inputDataSource = new DataSource()
      .withS3DataSource(new S3DataSource().withS3Uri(inputS3Path.toS3UriString)
        .withS3DataType(dataUploadResults match {
          case ObjectPrefixUploadResult(_) => S3DataType.S3Prefix.toString
          case ManifestDataUploadResult(_) => S3DataType.ManifestFile.toString
        })
        .withS3DataDistributionType(trainingS3DataDistribution))

    val inputChannel = new Channel()
      .withChannelName(trainingChannelName)
      .withCompressionType(trainingCompressionCodec.orNull)
      .withContentType(trainingContentType.orNull)
      .withDataSource(inputDataSource)
    createTrainingJobRequest.withInputDataConfig(inputChannel)

    val outputDataConfig = new OutputDataConfig()
      .withS3OutputPath(resolveS3Path(
        trainingOutputS3DataPath,
        trainingJobName,
        conf).toS3UriString)
      .withKmsKeyId(trainingKmsKeyId.orNull)
    createTrainingJobRequest.withOutputDataConfig(outputDataConfig)

    val resourceConfig = new ResourceConfig()
      .withInstanceCount(trainingInstanceCount)
      .withInstanceType(trainingInstanceType)
      .withVolumeSizeInGB(trainingInstanceVolumeSizeInGB)
    createTrainingJobRequest.withResourceConfig(resourceConfig)

    createTrainingJobRequest.withRoleArn(resolveRoleARN(sagemakerRole, conf).role)

    val stoppingCondition = new StoppingCondition()
      .withMaxRuntimeInSeconds(trainingMaxRuntimeInSeconds)
    createTrainingJobRequest.withStoppingCondition(stoppingCondition)

    createTrainingJobRequest
  }

  private def runTrainingJob(createTrainingJobRequest: CreateTrainingJobRequest,
                             trainingJobName: String): Unit = {
    val startingTrainingJobTime = this.timeProvider.currentTimeMillis

    try {
      sagemakerClient.createTrainingJob(createTrainingJobRequest)
      awaitTrainingCompletion(trainingJobName)
    } catch {
      case t: Throwable => throw new RuntimeException("Training job couldn't be completed.", t)
    } finally {
      val trainingJobTime = this.timeProvider.getElapsedTimeInSeconds(startingTrainingJobTime)
      log.info(s"Training Job Time: $trainingJobTime s")
    }
  }

  private def awaitTrainingCompletion(trainingJobName : String) : Unit = {
    val startTime = this.timeProvider.currentTimeMillis
    val describeTrainingJobRequest = new DescribeTrainingJobRequest()
      .withTrainingJobName(trainingJobName)
    InternalUtils.applyUserAgent(describeTrainingJobRequest)

    log.info(s"Begin waiting for training job $trainingJobName")
    while (this.timeProvider.currentTimeMillis - startTime < trainingJobTimeout.toMillis) {
      try {
        val response = sagemakerClient.describeTrainingJob(describeTrainingJobRequest)
        val currentStatus = TrainingJobStatus.fromValue(response.getTrainingJobStatus)
        log.info(s"Training job status: $currentStatus")
        currentStatus match {
          case TrainingJobStatus.Completed => return
          case TrainingJobStatus.Failed =>
            val message = s"Training job '$trainingJobName' failed for reason:" +
              s" '${response.getFailureReason}'"
            throw new RuntimeException(message)
          case TrainingJobStatus.Stopped =>
            val message = s"Training job '$trainingJobName' stopped. Stopping condition:" +
              s" '${response.getStoppingCondition}'"
            throw new RuntimeException(message)
          case _ => // for any other statuses, continue polling
        }
      } catch {
        case e : SdkBaseException =>
          if (!RetryUtils.isRetryableServiceException(e)) {
            throw e
          }
          log.warn(s"Retryable exception: ${e.getMessage}", e)
        case t : Throwable => throw t
      }
      timeProvider.sleep(SageMakerEstimator.TrainingJobPollInterval.toMillis)
    }
    throw new RuntimeException(s"Timed out after ${trainingJobTimeout.toString} while waiting for" +
      s" Training Job '$trainingJobName' to finish creating.")
  }

  private def deleteTrainingData(s3DataPath: S3DataPath) : Unit = {
    val s3Bucket = s3DataPath.bucket
    val s3Prefix = s3DataPath.objectPath

    try {
      val objectList = s3Client.listObjects(s3Bucket, s3Prefix)
      objectList.getObjectSummaries.forEach{
        s3Object => s3Client.deleteObject(s3Bucket, s3Object.getKey)
      }
      s3Client.deleteObject(s3Bucket, s3Prefix)
    } catch {
      case t: Throwable => log.warn(s"Received exception from s3 client. Data deletion failed. " +
        s"Stack trace: ${t.getStackTrace}")
    }
  }

  override def copy(extra: ParamMap): SageMakerEstimator = defaultCopy(extra)

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = schema

}
