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

import scala.util.{Failure, Success, Try}

import com.amazonaws.retry.RetryUtils

import com.amazonaws.services.sagemaker.AmazonSageMaker
import com.amazonaws.services.sagemaker.model._
import com.amazonaws.services.sagemaker.sparksdk.exceptions.{SageMakerResourceCleanupException, SageMakerSparkSDKException}
import com.amazonaws.services.sagemaker.sparksdk.internal.InternalUtils

/**
  * Deletes any SageMaker entities created during operation of the SageMaker
  * Estimator and Transformer.
  *
  * @param sagemaker The AWS SDK client used to make requests to SageMaker for deleting
  *                Training and Hosting resources.
  */
class SageMakerResourceCleanup(sagemaker : AmazonSageMaker) {

  /**
    * Deletes any SageMaker entities which have been created and still exist.
    * Any transient exceptions occurring from service calls will be rethrown wrapped in a
    * [[com.amazonaws.services.sagemaker.sparksdk.exceptions.SageMakerResourceCleanupException
    * SageMakerResourceCleanupException]] and can be retried.
    *
    * @param resources The set of entities which have been created.
    *                  Can be obtained from [[SageMakerModel.getCreatedResources]],
    *                  [[SageMakerEstimator.getCreatedResources]],
    *                  or any [[SageMakerSparkSDKException]] thrown during usage.
    */
  def deleteResources(resources: CreatedResources): Unit = {
    /* Only attempt to delete resources which were created during the execution of
     * the SageMaker Estimator/Model, to avoid accidentally deleting already existing resources
     * with the same name.
     *
     * For each resource which was created, first make a Describe call.
     * Only make the corresponding Delete call if the Describe call succeeded.
     */

    try {
      resources.endpointName.foreach(deleteEndpoint(_))
      resources.endpointConfigName.foreach(deleteEndpointConfig(_))
      resources.modelName.foreach(deleteModel(_))
    } catch {
      case t: Throwable => throw new SageMakerResourceCleanupException("Resource cleanup failed." +
        " Cleanup can be retried if appropriate by calling deleteResources again.", t, resources)
    }
  }

  def deleteEndpoint(name: String): Unit = {
    val describeEndpointRequest = new DescribeEndpointRequest().withEndpointName(name)
    InternalUtils.applyUserAgent(describeEndpointRequest)
    tryDescribe(sagemaker.describeEndpoint(describeEndpointRequest))
      .foreach(_ => {
        val deleteRequest = new DeleteEndpointRequest().withEndpointName(name)
        InternalUtils.applyUserAgent(deleteRequest)
        sagemaker.deleteEndpoint(deleteRequest)
      })
  }

  def deleteEndpointConfig(name: String): Unit = {
    val describeEndpointConfigRequest = new DescribeEndpointConfigRequest()
      .withEndpointConfigName(name)
    InternalUtils.applyUserAgent(describeEndpointConfigRequest)
    tryDescribe(sagemaker.describeEndpointConfig(describeEndpointConfigRequest))
      .foreach(_ => {
        val deleteRequest = new DeleteEndpointConfigRequest().withEndpointConfigName(name)
        InternalUtils.applyUserAgent(deleteRequest)
        sagemaker.deleteEndpointConfig(deleteRequest)
      })
  }

  def deleteModel(name: String): Unit = {
    val describeModelRequest = new DescribeModelRequest().withModelName(name)
    InternalUtils.applyUserAgent(describeModelRequest)
    tryDescribe(sagemaker.describeModel(describeModelRequest))
      .foreach(_ => {
        val deleteRequest = new DeleteModelRequest().withModelName(name)
        InternalUtils.applyUserAgent(deleteRequest)
        sagemaker.deleteModel(deleteRequest)
      })
  }

  /* Returns a Success if the entity exists, a Failure if it doesn't, or throws an Exception,
   * which indicates that we aren't sure, and that this can be retried. */
  private def tryDescribe[T](describeCall: => T): Try[T] = {
    Try(describeCall) match {
      case Success(v) => Success(v)
      case Failure(e) =>
        e match {
          case e : AmazonSageMakerException if !RetryUtils.isRetryableServiceException(e)
            => Failure(e)
          case t: Throwable => throw t
        }
    }
  }
}

/**
  * Resources that may have been created during operation of the SageMaker Estimator and
  * Model.
  *
  * @param modelName The name of the SageMaker Model that was created, or empty if it wasn't
  *                  created.
  * @param endpointConfigName The name of the SageMaker EndpointConfig that was
  *                           created, or empty if it wasn't created.
  * @param endpointName The name of the SageMaker Endpoint that was created, or empty
  *                     if it wasn't created.
  */
case class CreatedResources(modelName : Option[String],
                            endpointConfigName : Option[String],
                            endpointName : Option[String])
