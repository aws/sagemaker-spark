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

import com.amazonaws.AmazonServiceException
import org.mockito.{ArgumentCaptor, InOrder}
import org.mockito.Matchers.any
import org.mockito.Mockito.{inOrder, never, verify, when}
import org.scalatest.{BeforeAndAfter, FlatSpec}
import org.scalatest.mock.MockitoSugar
import org.scalatest.prop.TableDrivenPropertyChecks

import com.amazonaws.services.sagemaker.AmazonSageMaker
import com.amazonaws.services.sagemaker.model._
import com.amazonaws.services.sagemaker.sparksdk.exceptions.SageMakerResourceCleanupException

class SageMakerResourceCleanupTests
  extends FlatSpec with MockitoSugar with BeforeAndAfter with TableDrivenPropertyChecks {

  val resourcesWithAllCreated = CreatedResources(Option("m"), Option("ec"), Option("e"))
  val retryableException = new AmazonSageMakerException("msg")
  val nonValidationExceptions =
    Table(
      "Exception",
      retryableException,
      new AmazonServiceException("")
    )
  var sagemakerMock: AmazonSageMaker = _

  before {
    sagemakerMock = mock[AmazonSageMaker]
    cleanup = new SageMakerResourceCleanup(sagemakerMock)
    order = inOrder(sagemakerMock)

    // Return success by default for describe calls. Can be overridden by individual tests.
    when(sagemakerMock.describeEndpoint(any[DescribeEndpointRequest]))
      .thenReturn(new DescribeEndpointResult())
    when(sagemakerMock.describeEndpointConfig(any[DescribeEndpointConfigRequest]))
      .thenReturn(new DescribeEndpointConfigResult())
    when(sagemakerMock.describeModel(any[DescribeModelRequest]))
      .thenReturn(new DescribeModelResult())
  }

  "SageMakerResourceCleanup" should "pass correct name to DescribeEndpoint" in {
    cleanup.deleteResources(resourcesWithAllCreated)

    val describeEndpointCaptor = ArgumentCaptor.forClass(classOf[DescribeEndpointRequest])
    verify(sagemakerMock).describeEndpoint(describeEndpointCaptor.capture)
    assert(describeEndpointCaptor.getValue.getEndpointName == resourcesWithAllCreated
      .endpointName.get)
  }

  it should "pass correct name to DescribeEndpointConfig" in {
    cleanup.deleteResources(resourcesWithAllCreated)

    val describeEndpointConfigCaptor = ArgumentCaptor
      .forClass(classOf[DescribeEndpointConfigRequest])
    verify(sagemakerMock).describeEndpointConfig(describeEndpointConfigCaptor.capture)
    assert(describeEndpointConfigCaptor.getValue.getEndpointConfigName == resourcesWithAllCreated
      .endpointConfigName.get)
  }

  it should "pass correct name to DescribeModel" in {
    cleanup.deleteResources(resourcesWithAllCreated)

    val describeModelCaptor = ArgumentCaptor.forClass(classOf[DescribeModelRequest])
    verify(sagemakerMock).describeModel(describeModelCaptor.capture)
    assert(describeModelCaptor.getValue.getModelName == resourcesWithAllCreated.modelName.get)
  }

  it should "pass correct name to DeleteEndpoint" in {
    cleanup.deleteResources(resourcesWithAllCreated)

    val deleteEndpointCaptor = ArgumentCaptor.forClass(classOf[DeleteEndpointRequest])
    verify(sagemakerMock).deleteEndpoint(deleteEndpointCaptor.capture)
    assert(deleteEndpointCaptor.getValue.getEndpointName == resourcesWithAllCreated.endpointName
      .get)
  }

  it should "pass correct name to DeleteEndpointConfig" in {
    cleanup.deleteResources(resourcesWithAllCreated)

    val deleteEndpointConfigCaptor = ArgumentCaptor.forClass(classOf[DeleteEndpointConfigRequest])
    verify(sagemakerMock).deleteEndpointConfig(deleteEndpointConfigCaptor.capture)
    assert(deleteEndpointConfigCaptor.getValue.getEndpointConfigName == resourcesWithAllCreated
      .endpointConfigName.get)
  }

  it should "pass correct name to DeleteModel" in {
    cleanup.deleteResources(resourcesWithAllCreated)

    val deleteModelCaptor = ArgumentCaptor.forClass(classOf[DeleteModelRequest])
    verify(sagemakerMock).deleteModel(deleteModelCaptor.capture)
    assert(deleteModelCaptor.getValue.getModelName == resourcesWithAllCreated.modelName.get)
  }

  it should "make Describe and Delete calls in the correct order" in {
    cleanup.deleteResources(resourcesWithAllCreated)

    /* We should delete resources in the reverse order that they were created,
     * since Endpoint depends on EndpointConfig, and EndpointConfig depends on Model.
     * For each resource, only do the deletion after a successful describe call. */
    order.verify(sagemakerMock).describeEndpoint(any[DescribeEndpointRequest])
    order.verify(sagemakerMock).deleteEndpoint(any[DeleteEndpointRequest])
    order.verify(sagemakerMock).describeEndpointConfig(any[DescribeEndpointConfigRequest])
    order.verify(sagemakerMock).deleteEndpointConfig(any[DeleteEndpointConfigRequest])
    order.verify(sagemakerMock).describeModel(any[DescribeModelRequest])
    order.verify(sagemakerMock).deleteModel(any[DeleteModelRequest])
  }

  it should "not make any Describe or Delete calls for resources that weren't created" in {
    val resources = CreatedResources(Option.empty, Option.empty, Option.empty)
    cleanup.deleteResources(resources)

    verify(sagemakerMock, never).describeEndpoint(any[DescribeEndpointRequest])
    verify(sagemakerMock, never).describeEndpointConfig(any[DescribeEndpointConfigRequest])
    verify(sagemakerMock, never).describeModel(any[DescribeModelRequest])

    verify(sagemakerMock, never).deleteEndpoint(any[DeleteEndpointRequest])
    verify(sagemakerMock, never).deleteEndpointConfig(any[DeleteEndpointConfigRequest])
    verify(sagemakerMock, never).deleteModel(any[DeleteModelRequest])
  }

  it should "not make a Delete call if the corresponding Describe call indicates it does not " +
    "exist" in {
    when(sagemakerMock.describeEndpoint(any[DescribeEndpointRequest]))
      .thenThrow(new AmazonSageMakerException("does not exist"))
    when(sagemakerMock.describeEndpointConfig(any[DescribeEndpointConfigRequest]))
      .thenThrow(new AmazonSageMakerException("does not exist"))
    when(sagemakerMock.describeModel(any[DescribeModelRequest]))
      .thenThrow(new AmazonSageMakerException("does not exist"))
    cleanup.deleteResources(resourcesWithAllCreated)

    // Describe calls should be made
    verify(sagemakerMock).describeEndpoint(any[DescribeEndpointRequest])
    verify(sagemakerMock).describeEndpointConfig(any[DescribeEndpointConfigRequest])
    verify(sagemakerMock).describeModel(any[DescribeModelRequest])

    // No delete calls should be made
    verify(sagemakerMock, never).deleteEndpoint(any[DeleteEndpointRequest])
    verify(sagemakerMock, never).deleteEndpointConfig(any[DeleteEndpointConfigRequest])
    verify(sagemakerMock, never).deleteModel(any[DeleteModelRequest])
  }
  var cleanup: SageMakerResourceCleanup = _
  retryableException.setStatusCode(500)
  var order: InOrder = _

  it should "rethrow any exceptions other than ValidationErrorException from DescribeEndpoint " +
    "wrapped in SageMakerResourceCleanupException" in {
    forAll(nonValidationExceptions) { (e: Exception) =>
      val sagemakerMock = mock[AmazonSageMaker]
      val cleanup = new SageMakerResourceCleanup(sagemakerMock)
      when(sagemakerMock.describeEndpoint(any[DescribeEndpointRequest])).thenThrow(e)

      val caught =
        intercept[SageMakerResourceCleanupException] {
          cleanup.deleteResources(resourcesWithAllCreated)
        }
      assert(e.getClass == caught.cause.getClass)
      verify(sagemakerMock, never).deleteEndpoint(any[DeleteEndpointRequest])
    }
  }

  it should "rethrow any exceptions other than ValidationErrorException from " +
    "DescribeEndpointConfig wrapped in SageMakerResourceCleanupException" in {
    forAll(nonValidationExceptions) { (e: Exception) =>
      val sagemakerMock = mock[AmazonSageMaker]
      val cleanup = new SageMakerResourceCleanup(sagemakerMock)
      when(sagemakerMock.describeEndpointConfig(any[DescribeEndpointConfigRequest])).thenThrow(e)

      val caught =
        intercept[SageMakerResourceCleanupException] {
          cleanup.deleteResources(resourcesWithAllCreated)
        }
      assert(e.getClass == caught.cause.getClass)
      verify(sagemakerMock, never).deleteEndpointConfig(any[DeleteEndpointConfigRequest])
    }
  }

  it should "rethrow any exceptions other than ValidationErrorException from DescribeModel " +
    "wrapped in SageMakerResourceCleanupException" in {
    forAll(nonValidationExceptions) { (e: Exception) =>
      val sagemakerMock = mock[AmazonSageMaker]
      val cleanup = new SageMakerResourceCleanup(sagemakerMock)
      when(sagemakerMock.describeModel(any[DescribeModelRequest])).thenThrow(e)

      val caught =
        intercept[SageMakerResourceCleanupException] {
          cleanup.deleteResources(resourcesWithAllCreated)
        }
      assert(e.getClass == caught.cause.getClass)
      verify(sagemakerMock, never).deleteModel(any[DeleteModelRequest])
    }
  }
}
