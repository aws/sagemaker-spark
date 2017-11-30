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

import scala.collection.JavaConverters.mapAsJavaMapConverter

import com.amazonaws.SdkBaseException
import com.amazonaws.retry.RetryUtils

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.Model
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.types.StructType

import com.amazonaws.services.sagemaker.{AmazonSageMaker, AmazonSageMakerClientBuilder}
import com.amazonaws.services.sagemaker.model._
import com.amazonaws.services.sagemaker.sparksdk.EndpointCreationPolicy.EndpointCreationPolicy
import com.amazonaws.services.sagemaker.sparksdk.exceptions.SageMakerSparkSDKException
import com.amazonaws.services.sagemaker.sparksdk.internal.{InternalUtils, SystemTimeProvider, TimeProvider}
import com.amazonaws.services.sagemaker.sparksdk.transformation.{RequestRowSerializer, ResponseRowDeserializer}
import com.amazonaws.services.sagemaker.sparksdk.transformation.deserializers.LibSVMResponseRowDeserializer
import com.amazonaws.services.sagemaker.sparksdk.transformation.serializers.LibSVMRequestRowSerializer
import com.amazonaws.services.sagemaker.sparksdk.transformation.util.{RequestBatchIterator, RequestBatchIteratorFactory}


object SageMakerModel {
  var EndpointCreationTimeout : Duration = Duration.ofMinutes(30)
  var EndpointPollInterval : Duration = Duration.ofSeconds(5)

  /**
    * Creates a SageMakerModel that can be used to transform DataFrames from a given successfully
    * completed Training Job.
    *
    * @param trainingJobName The name of the successfully completed training job.
    * @param modelImage The URI of the image that will serve model inferences.
    * @param modelExecutionRoleARN The IAM Role used by SageMaker when running the hosted Model and
    *                              to download model data from S3.
    * @param endpointInstanceType The instance type used to run the model container.
    * @param endpointInitialInstanceCount The minimum number of instances used to host the model.
    * @param requestRowSerializer Serializes a Row to an Array of Bytes.
    * @param responseRowDeserializer Deserializes an Array of Bytes to a series of Rows.
    * @param modelEnvironmentVariables The environment variables that SageMaker will set on the
    *                                  model container during execution.
    * @param endpointCreationPolicy Whether the endpoint is created upon SageMakerModel
    *                               construction, transformation, or not at all.
    * @param sagemakerClient Amazon SageMaker client. Used to send CreateTrainingJob, CreateModel,
    *                        and CreateEndpoint requests.
    * @param prependResultRows Whether the transformation result should also include the input Rows.
    *                          If true, each output Row is formed by a concatenation of the input
    *                          Row with the corresponding Row produced by SageMaker invocation,
    *                          produced by responseRowDeserializer. If false, each output Row is
    *                          just taken
    *                          from responseRowDeserializer.
    * @param namePolicy The NamePolicy to use when naming SageMaker entities created during usage of
    *                   this Model.
    * @param uid The unique identifier of this Estimator. Used to represent this stage in
    *            Spark ML pipelines.
    * @return A [[SageMakerModel]] that sends InvokeEndpoint requests to an endpoint hosting
    *         the training job's model.
    */
  def fromTrainingJob(trainingJobName: String,
                      modelImage: String,
                      modelExecutionRoleARN: String,
                      endpointInstanceType: String,
                      endpointInitialInstanceCount : Int,
                      requestRowSerializer: RequestRowSerializer,
                      responseRowDeserializer: ResponseRowDeserializer,
                      modelEnvironmentVariables: Map[String, String] = Map[String, String](),
                      endpointCreationPolicy: EndpointCreationPolicy =
                        EndpointCreationPolicy.CREATE_ON_CONSTRUCT,
                      sagemakerClient : AmazonSageMaker
                        = AmazonSageMakerClientBuilder.defaultClient,
                      prependResultRows : Boolean = true,
                      namePolicy : NamePolicy = new RandomNamePolicy(),
                      uid: String = Identifiable.randomUID("sagemaker")) : SageMakerModel = {
    require(endpointCreationPolicy != EndpointCreationPolicy.DO_NOT_CREATE,
      "Endpoint creation policy must not be DO_NOT_CREATE to" +
        " create an endpoint from a training job name.")

    val describeTrainingJobRequest = new DescribeTrainingJobRequest()
      .withTrainingJobName(trainingJobName)
    InternalUtils.applyUserAgent(describeTrainingJobRequest)
    val response = sagemakerClient.describeTrainingJob(describeTrainingJobRequest)

    val status = TrainingJobStatus.fromValue(response.getTrainingJobStatus)
    require(status == TrainingJobStatus.Completed || status == TrainingJobStatus.Stopped,
      "Can only create a SageMakerModel from a training job with status" +
        " Completed or Stopped, not status " + status.toString)

    val modelPath = response.getModelArtifacts.getS3ModelArtifacts
    new SageMakerModel(modelImage = Some(modelImage),
      modelPath = Some(S3DataPath.fromS3URI(modelPath)),
      requestRowSerializer = requestRowSerializer,
      responseRowDeserializer = responseRowDeserializer,
      modelEnvironmentVariables = modelEnvironmentVariables,
      modelExecutionRoleARN = Some(modelExecutionRoleARN),
      endpointCreationPolicy = endpointCreationPolicy,
      endpointInstanceType = Some(endpointInstanceType),
      endpointInitialInstanceCount = Some(endpointInitialInstanceCount),
      sagemakerClient = sagemakerClient,
      prependResultRows = prependResultRows,
      namePolicy = namePolicy,
      uid = uid)
  }

  /**
    * Creates a SageMakerModel that can be used to transform DataFrames using a given model stored
    * in S3.
    *
    * @param modelPath The S3 URI to the model data to host.
    * @param modelImage The URI of the image that will serves model inferences.
    * @param modelExecutionRoleARN The IAM Role used by SageMaker when running the hosted Model and
    *                              to download model data from S3.
    * @param requestRowSerializer Serializes a Row to an Array of Bytes.
    * @param responseRowDeserializer Deserializes an Array of Bytes to a series of Rows.
    * @param modelEnvironmentVariables The environment variables that SageMaker will set on the
    *                                  model container during execution.
    * @param endpointInstanceType The instance type used to run the model container.
    * @param endpointInitialInstanceCount The minimum number of instances used to host the model.
    * @param endpointCreationPolicy Whether the endpoint is created upon SageMakerModel
    *                               construction, transformation, or not at all.
    * @param sagemakerClient Amazon SageMaker client. Used to send CreateTrainingJob, CreateModel,
    *                        and CreateEndpoint requests.
    * @param prependResultRows Whether the transformation result should also include the input Rows.
    *                          If true, each output Row is formed by a concatenation of the input
    *                          Row with the corresponding Row produced by SageMaker invocation,
    *                          produced by responseRowDeserializer. If false, each output Row is
    *                          just taken
    *                          from responseRowDeserializer.
    * @param namePolicy The NamePolicy to use when naming SageMaker entities created during usage of
    *                   this Model.
    * @param uid The unique identifier of this Estimator. Used to represent this stage in
    *            Spark ML pipelines.
    * @return A [[SageMakerModel]] that sends InvokeEndpoint requests to an endpoint hosting
    *         the given model.
    */
  def fromModelS3Path(modelPath: String,
                      modelImage: String,
                      modelExecutionRoleARN: String,
                      endpointInstanceType: String,
                      endpointInitialInstanceCount : Int,
                      requestRowSerializer: RequestRowSerializer,
                      responseRowDeserializer: ResponseRowDeserializer,
                      modelEnvironmentVariables: Map[String, String] = Map[String, String](),
                      endpointCreationPolicy: EndpointCreationPolicy =
                        EndpointCreationPolicy.CREATE_ON_CONSTRUCT,
                      sagemakerClient : AmazonSageMaker
                        = AmazonSageMakerClientBuilder.defaultClient,
                      prependResultRows : Boolean = true,
                      namePolicy : NamePolicy = new RandomNamePolicy(),
                      uid: String = Identifiable.randomUID("sagemaker")) : SageMakerModel = {
    require(endpointCreationPolicy != EndpointCreationPolicy.DO_NOT_CREATE,
      "Endpoint creation policy must not be DO_NOT_CREATE to create an endpoint from a model path.")

    new SageMakerModel(modelImage = Some(modelImage),
      modelPath = Some(S3DataPath.fromS3URI(modelPath)),
      requestRowSerializer = requestRowSerializer,
      responseRowDeserializer = responseRowDeserializer,
      modelEnvironmentVariables = modelEnvironmentVariables,
      modelExecutionRoleARN = Some(modelExecutionRoleARN),
      endpointCreationPolicy = endpointCreationPolicy,
      endpointInstanceType = Some(endpointInstanceType),
      endpointInitialInstanceCount = Some(endpointInitialInstanceCount),
      sagemakerClient = sagemakerClient,
      prependResultRows = prependResultRows,
      namePolicy = namePolicy,
      uid = uid)
  }

  /**
    * Creates a SageMakerModel that can be used to transform DataFrames based on endpointName.
    *
    * @param endpointName The name of an endpoint that is current in service.
    * @param requestRowSerializer Serializes a Row to an Array of Bytes
    * @param responseRowDeserializer Deserializes an Array of Bytes to a series of Rows
    * @param modelEnvironmentVariables The environment variables that SageMaker will set on the
    *                                  model container during execution.
    * @param sagemakerClient Amazon SageMaker client. Used to send CreateTrainingJob, CreateModel,
    *                        and CreateEndpoint requests.
    * @param prependResultRows Whether the transformation result should also include the input Rows.
    *                          If true, each output Row is formed by a concatenation of the input
    *                          Row with the corresponding Row produced by SageMaker invocation,
    *                          produced by responseRowDeserializer. If false, each output Row is
    *                          just taken from responseRowDeserializer.
    * @param namePolicy The NamePolicy to use when naming SageMaker entities created during usage of
    *                   this Model.
    * @param uid The unique identifier of this Estimator. Used to represent this stage in
    *            Spark ML pipelines.
    * @return a [[SageMakerModel]] that sends InvokeEndpoint requests to the given endpoint.
    */
  def fromEndpoint(endpointName: String,
                   requestRowSerializer: RequestRowSerializer,
                   responseRowDeserializer: ResponseRowDeserializer,
                   modelEnvironmentVariables: Map[String, String] = Map[String, String](),
                   sagemakerClient : AmazonSageMaker
                   = AmazonSageMakerClientBuilder.defaultClient,
                   prependResultRows : Boolean = true,
                   namePolicy : NamePolicy = new RandomNamePolicy(),
                   uid: String = Identifiable.randomUID("sagemaker")) : SageMakerModel = {

    val describeEndpointRequest = new DescribeEndpointRequest().withEndpointName(endpointName)
    InternalUtils.applyUserAgent(describeEndpointRequest)
    val result = sagemakerClient.describeEndpoint(describeEndpointRequest)

    val status = result.getEndpointStatus

    require(status == EndpointStatus.InService.toString, "Endpoint status must be In Service, not "
      + status)

    new SageMakerModel(existingEndpointName = Some(endpointName),
      requestRowSerializer = requestRowSerializer,
      responseRowDeserializer = responseRowDeserializer,
      modelEnvironmentVariables = modelEnvironmentVariables,
      endpointCreationPolicy = EndpointCreationPolicy.DO_NOT_CREATE,
      endpointInstanceType = None,
      endpointInitialInstanceCount = None,
      sagemakerClient = sagemakerClient,
      prependResultRows = prependResultRows,
      namePolicy = namePolicy,
      uid = uid)
  }

}

/**
  * A Model implementation which transforms a DataFrame by making requests to a SageMaker Endpoint.
  * Manages life cycle of all necessary SageMaker entities, including Model, EndpointConfig, and
  * Endpoint.
  *
  * This Model transforms one DataFrame to another by repeated, distributed SageMaker Endpoint
  * invocation.
  * Each invocation request body is formed by concatenating input DataFrame Rows serialized to
  * Byte Arrays by the specified
  * [[com.amazonaws.services.sagemaker.sparksdk.transformation.RequestRowSerializer serializer]].
  * The invocation request content-type property is set from
  * [[com.amazonaws.services.sagemaker.sparksdk.transformation.RequestRowSerializer.contentType
  * contentType]].
  * The invocation request accepts property is set from the deserializer's
  * [[com.amazonaws.services.sagemaker.sparksdk.transformation.ResponseRowDeserializer.accepts
  * accepts]].
  *
  * The transformed DataFrame is produced by deserializing each invocation response body into a
  * series of Rows. Row deserialization is delegated to the specified
  * [[com.amazonaws.services.sagemaker.sparksdk.transformation.ResponseRowDeserializer
  * deserializer]], which converts an Array of Bytes to an Iterator[Row]. If
  * prependInputRows is false, the transformed DataFrame will contain just these Rows. If
  * prependInputRows is true, then each transformed Row is a concatenation of the input Row with its
  * corresponding SageMaker invocation deserialized Row.
  *
  * Each invocation of [[SageMakerModel!.transform* transform]] passes the [[Dataset.schema]] of
  * the input DataFrame to requestRowSerialize by invoking
  * [[com.amazonaws.services.sagemaker.sparksdk.transformation.RequestRowSerializer.setSchema
  * setSchema]].
  *
  * The specified [[com.amazonaws.services.sagemaker.sparksdk.transformation.RequestRowSerializer
  * serializer]] also controls the validity of input Row Schemas for this
  * Model. Schema validation is carried out on each call to [[SageMakerModel!.transformSchema
  * transformSchema]], which invokes
  * [[com.amazonaws.services.sagemaker.sparksdk.transformation.RequestRowSerializer.validateSchema
  * validateSchema]].
  *
  * Adapting this SageMaker model to the data format and type of a specific Endpoint is achieved by
  * sub-classing [[com.amazonaws.services.sagemaker.sparksdk.transformation.RequestRowSerializer
  * RequestRowSerializer]] and
  * [[com.amazonaws.services.sagemaker.sparksdk.transformation.ResponseRowDeserializer
  * RequestRowDeserializer]].
  * Examples of a Serializer and Deseralizer are
  * [[com.amazonaws.services.sagemaker.sparksdk.transformation.serializers.
  * LibSVMRequestRowSerializer]] and
  * [[com.amazonaws.services.sagemaker.sparksdk.transformation.deserializers.
  * LibSVMResponseRowDeserializer]]
  * respectively.
  *
  * @param endpointInstanceType The instance type used to run the model container.
  * @param endpointInitialInstanceCount The minimum number of instances used to host the model.
  * @param requestRowSerializer Serializes a Row to an Array of Bytes.
  * @param responseRowDeserializer Deserializes an Array of Bytes to a series of Rows.
  * @param existingEndpointName An endpoint name.
  * @param modelImage A Docker image URI.
  * @param modelPath An S3 location that a successfully completed SageMaker Training Job has stored
  *                  its model output to.
  * @param modelEnvironmentVariables The environment variables that SageMaker will set on the model
  *                                  container during execution.
  * @param modelExecutionRoleARN The IAM Role used by SageMaker when running the hosted Model and to
  *                              download model data from S3.
  * @param endpointCreationPolicy Whether the endpoint is created upon SageMakerModel construction,
  *                               transformation, or not at all.
  * @param sagemakerClient Amazon SageMaker client. Used to send CreateTrainingJob, CreateModel,
  *                        and CreateEndpoint requests.
  * @param prependResultRows Whether the transformation result should also include the input Rows.
  *                          If true, each output Row is formed by a concatenation of the input
  *                          Row with the corresponding Row produced by SageMaker invocation,
  *                          produced by responseRowDeserializer. If false, each output Row is
  *                          just taken from responseRowDeserializer.
  * @param namePolicy The NamePolicy to use when naming SageMaker entities created during usage of
  *                   this Model.
  * @param uid The unique identifier of this Estimator. Used to represent this stage in
  *            Spark ML pipelines.
  */
class SageMakerModel(
                     val endpointInstanceType : Option[String],
                     val endpointInitialInstanceCount : Option[Int],
                     val requestRowSerializer : RequestRowSerializer,
                     val responseRowDeserializer: ResponseRowDeserializer,
                     val existingEndpointName : Option[String] = Option.empty,
                     val modelImage : Option[String] = Option.empty,
                     val modelPath : Option[S3DataPath] = Option.empty,
                     val modelEnvironmentVariables : Map[String, String] =
                      Map[String, String](),
                     val modelExecutionRoleARN : Option[String] = Option.empty,
                     val endpointCreationPolicy : EndpointCreationPolicy =
                      EndpointCreationPolicy.CREATE_ON_CONSTRUCT,
                     val sagemakerClient : AmazonSageMaker
                       = AmazonSageMakerClientBuilder.defaultClient,
                     val prependResultRows : Boolean = true,
                     val namePolicy : NamePolicy = new RandomNamePolicy(),
                     override val uid: String = Identifiable.randomUID("sagemaker"))
  extends Model[SageMakerModel] {

  if (existingEndpointName.nonEmpty) {
    require(endpointCreationPolicy == EndpointCreationPolicy.DO_NOT_CREATE,
    "SageMakerModel's endpoint creation policy must be DO_NOT_CREATE if creating an " +
      "SageMakerModel from an existing endpoint.")
  }

  private[sparksdk] var timeProvider : TimeProvider = new SystemTimeProvider()

  /* These mutable fields keep track of the SageMaker resources which have been
   *  successfully created so far, so they can be deleted later. */
  private var createdModelName : Option[String] = Option.empty
  private var createdEndpointConfigName : Option[String] = Option.empty
  private var createdEndpointName : Option[String] = Option.empty

  if (endpointCreationPolicy == EndpointCreationPolicy.CREATE_ON_CONSTRUCT) {
    createSageMakerHostingResources()
  }

  /**
    * An endpoint name if it exists or [[None]] otherwise.
    * @return An endpoint name.
    */
  def endpointName: Option[String] = {
    if (existingEndpointName.isEmpty && createdEndpointName.isEmpty) {
      None
    } else {
      Some(existingEndpointName.getOrElse(createdEndpointName.get))
    }
  }

  private def createSageMakerHostingResources() : Unit = {
    val startingEndpointCreationTime = this.timeProvider.currentTimeMillis
    try {
      createdModelName = Option(createModel)
      createdEndpointConfigName = Option(createEndpointConfig)
      createdEndpointName = Option(createEndpoint)
      awaitEndpointCompletion(createdEndpointName.get)

    } catch {
      /* For certain exceptions, most notably timeout exceptions, we can't be sure if the resource
       * was created successfully on the server side. In these cases, we will still treat the
       * resource as not having been created, to err on the side of not deleting in this
       * ambiguous case. */
      case t: Throwable => throw SageMakerSparkSDKException(
        "SageMaker resource creation failed.", t, getCreatedResources)
    } finally {
      val endpointCreationTime = this.timeProvider
        .getElapsedTimeInSeconds(startingEndpointCreationTime)
      log.info(s"Endpoint Creation Time: $endpointCreationTime s")
    }
  }

  /**
    * Gets potentially created resources during operation of the SageMakerModel.
    * @return Resources that may have been created.
    */
  def getCreatedResources: CreatedResources = {
    CreatedResources(this.createdModelName,
                     this.createdEndpointConfigName,
                     endpointName)
  }

  private def createModel : String = {
    val modelName = this.namePolicy.modelName
    val containerDefinition = new ContainerDefinition()
        .withContainerHostname(modelName)
        .withImage(modelImage.get)
        .withEnvironment(modelEnvironmentVariables.asJava)
        .withModelDataUrl(this.modelPath.get.toS3UriString)
    val request = new CreateModelRequest()
      .withModelName(modelName)
      .withPrimaryContainer(containerDefinition)
      .withExecutionRoleArn(modelExecutionRoleARN.orNull)
    InternalUtils.applyUserAgent(request)
    log.info(s"Create model request: ${request.toString}")
    sagemakerClient.createModel(request)
    log.info(s"Created model $modelName")
    modelName
  }

  /* We only need to use one variant for the endpoint, since we don't need A/B testing
   * or shadow mode features. */
  private final val VariantName = "AllRequests"

  private def createEndpointConfig : String = {
    val endpointConfigName = this.namePolicy.endpointConfigName
    val productionVariant = new ProductionVariant()
        .withModelName(this.namePolicy.modelName)
        .withInstanceType(endpointInstanceType.get)
        .withInitialInstanceCount(endpointInitialInstanceCount.get)
        .withVariantName(VariantName)
    val request = new CreateEndpointConfigRequest()
      .withEndpointConfigName(endpointConfigName)
      .withProductionVariants(productionVariant)
      .withEndpointConfigName(this.namePolicy.endpointConfigName)
    InternalUtils.applyUserAgent(request)
    log.info(s"Create EndpointConfig request: ${request.toString}")
    sagemakerClient.createEndpointConfig(request)
    log.info(s"Created endpoint config $endpointConfigName")
    endpointConfigName
  }

  private def createEndpoint : String = {
    val endpointName = this.namePolicy.endpointName
    val request = new CreateEndpointRequest()
      .withEndpointConfigName(this.namePolicy.endpointConfigName)
      .withEndpointName(endpointName)
    InternalUtils.applyUserAgent(request)
    log.info(s"Create Endpoint request: ${request.toString}")
    sagemakerClient.createEndpoint(request)
    log.info(s"Created endpoint $endpointName")
    endpointName
  }

  private def awaitEndpointCompletion(endpointName : String) : Unit = {
    val startTime = this.timeProvider.currentTimeMillis
    val describeEndpointRequest = new DescribeEndpointRequest().withEndpointName(endpointName)
    InternalUtils.applyUserAgent(describeEndpointRequest)
    log.info(s"Begin waiting for endpoint completion for endpoint $endpointName")
    while (this.timeProvider.currentTimeMillis - startTime <
      SageMakerModel.EndpointCreationTimeout.toMillis) {
      try {
        val response = sagemakerClient.describeEndpoint(describeEndpointRequest)
        val currentStatus = EndpointStatus.fromValue(response.getEndpointStatus)
        log.info(s"Endpoint creation status: $currentStatus")
        currentStatus match {
          case EndpointStatus.InService => return
          case EndpointStatus.Failed =>
            val message = s"Endpoint '$endpointName' failed for reason:" +
              s" '${response.getFailureReason}'"
            throw new RuntimeException(message)
          case _ => // for any other statuses, continue polling
        }
      } catch {
        case e : SdkBaseException =>
          if (!RetryUtils.isRetryableServiceException(e)) {
            throw e
          }
          log.warn(s"Retryable exception: ${e.getMessage}")
        case t : Throwable => throw t
      }

      timeProvider.sleep(SageMakerModel.EndpointPollInterval.toMillis)
    }

    throw new RuntimeException(
      s"Timed out after ${SageMakerModel.EndpointCreationTimeout.toString}" +
      s" while waiting for Endpoint '$endpointName' to finish creating.")
  }

  /**
    * Transforms the input dataset.
    *
    * Transforms Dataset to DataFrame by repeated, distributed SageMaker Endpoint invocation
    * using [[RequestBatchIterator]].
    * Creates all necessary SageMaker entities if specified by the [[EndpointCreationPolicy]]
    * and if an Endpoint doesn't exist yet.
    * @param dataset An input dataset.
    * @return Transformed dataset.
    */
  override def transform(dataset: Dataset[_]): DataFrame = {
    if (endpointCreationPolicy == EndpointCreationPolicy.CREATE_ON_TRANSFORM &&
        endpointName.isEmpty) {
      createSageMakerHostingResources()
    }
    val dataFrame = dataset.toDF()
    requestRowSerializer.setSchema(dataFrame.schema)
    val resultSchema = transformSchema(dataFrame.schema)

    require(endpointName.nonEmpty, "Endpoint name is empty. " +
      "Create an endpoint to set the endpoint name.")
    val resultDF = dataFrame.mapPartitions(RequestBatchIteratorFactory.createRequestBatchIterator
      (endpointName.get, requestRowSerializer,
        responseRowDeserializer, prependResultRows))(RowEncoder(resultSchema))

    resultDF
  }

  /**
    * Checks transform validity of the input schema and provides the output schema.
    *
    * Validates the input schema against
    * [[com.amazonaws.services.sagemaker.sparksdk.transformation.RequestRowSerializer
    * RequestRowSerializer]] and returns
    * [[com.amazonaws.services.sagemaker.sparksdk.transformation.ResponseRowDeserializer
    * ResponseRowDeserializer]] schema.
    * Prepends the output with the input schema if required by the SageMakerModel.
    *
    * @param schema Input schema to be validated and transformed.
    * @return Output schema
    */
  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = {
    requestRowSerializer.validateSchema(schema)

    if (prependResultRows) {
      new StructType((schema ++ responseRowDeserializer.schema).seq.toArray)
    } else {
      responseRowDeserializer.schema
    }
  }

  /**
    * Creates a copy of the SageMakerModel instance with the same instance variables
    * and extra params.
    *
    * Sets the EndpointCreationPolicy to EndpointCreationPolicy.DO_NOT_CREATE
    * regardless of the original EndpointCreationPolicy value so that
    * copies do not make Endpoints.
    *
    * @param extra Params to be applied to the new instance
    * @return The copy of SageMakerModel
    */
  override def copy(extra: ParamMap): SageMakerModel = {
    val that = this.getClass.getConstructor(
      classOf[Option[String]],
      classOf[Option[Int]],
      classOf[RequestRowSerializer],
      classOf[ResponseRowDeserializer],
      classOf[Option[String]],
      classOf[Option[String]],
      classOf[Option[String]],
      classOf[Map[String, String]],
      classOf[Option[String]],
      classOf[EndpointCreationPolicy],
      classOf[AmazonSageMaker],
      classOf[Boolean],
      classOf[NamePolicy],
      classOf[String]).newInstance(
      this.endpointInstanceType,
      this.endpointInitialInstanceCount,
      this.requestRowSerializer,
      this.responseRowDeserializer,
      this.endpointName,
      this.modelImage,
      this.modelPath,
      this.modelEnvironmentVariables,
      this.modelExecutionRoleARN,
      EndpointCreationPolicy.DO_NOT_CREATE,
      this.sagemakerClient,
      Boolean.box(this.prependResultRows),
      this.namePolicy,
      this.uid
    )
    copyValues(that, extra)
  }
}
