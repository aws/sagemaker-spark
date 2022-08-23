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

import java.nio.ByteBuffer
import java.util
import java.util.NoSuchElementException

import com.amazonaws.{AmazonWebServiceRequest, ResponseMetadata}
import com.amazonaws.regions.Region
import com.amazonaws.services.sagemakerruntime.AmazonSageMakerRuntime
import com.amazonaws.services.sagemakerruntime.model.{InvokeEndpointAsyncRequest, InvokeEndpointAsyncResult, InvokeEndpointRequest, InvokeEndpointResult}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}
import org.scalatest.mock.MockitoSugar

import org.apache.spark.sql.Row
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.types.{IntegerType, StructField, StructType}

import com.amazonaws.services.sagemaker.sparksdk.transformation.{RequestRowSerializer, ResponseRowDeserializer}
import com.amazonaws.services.sagemaker.sparksdk.transformation.util.RequestBatchIterator

class RequestBatchIteratorTests extends FlatSpec with Matchers with MockitoSugar with
  BeforeAndAfter {

  /**
    * An AmazonSageMakerRuntime that reads a series of integers and returns a new series
    * consisting of the original integers multiplied by two
    */
  class TestSageMakerTransformer extends AmazonSageMakerRuntime {
    override def invokeEndpoint(invokeEndpointRequest: InvokeEndpointRequest):
    InvokeEndpointResult = {
      val requestBody = invokeEndpointRequest.getBody
      val byteBuffer = ByteBuffer.allocate(100)
      while(requestBody.hasRemaining) {
        byteBuffer.putInt(requestBody.getInt() * 2)
      }
      byteBuffer.flip()
      new InvokeEndpointResult()
         .withBody(byteBuffer)
         .withContentType(invokeEndpointRequest.getContentType)
    }
    override def invokeEndpointAsync(invokeEndpointAsyncRequest: InvokeEndpointAsyncRequest):
    InvokeEndpointAsyncResult = {
      val inputLocation = invokeEndpointAsyncRequest.getInputLocation
      val invokeEndpointAsyncResult = new InvokeEndpointAsyncResult()
      invokeEndpointAsyncResult.setOutputLocation(inputLocation)
      return invokeEndpointAsyncResult
    }
    override def shutdown(): Unit = {}
    override def getCachedResponseMetadata(request: AmazonWebServiceRequest): ResponseMetadata =
    new ResponseMetadata(new util.HashMap[String, String]())

  }

  val emptyRequestObjectSerializer = new RequestRowSerializer {
    override def serializeRow(obj: Row): Array[Byte] = new Array[Byte](0)
    override val contentType: String = "None"
  }

  val emptyResponseObjectsDeserializer = new ResponseRowDeserializer {
    override def deserializeResponse(inputData: Array[Byte]): Iterator[Row] =
      Seq.empty[Row].iterator
    override val accepts: String = "None"
    override val schema: StructType = new StructType()
  }

  val emptySageMakerTransformer = new TestSageMakerTransformer ()
  RequestBatchIterator.sagemakerRuntime = emptySageMakerTransformer

  "RequestBatchIterator" should "not accept batch size constraints that are less than 1" in {
    intercept[IllegalArgumentException]{
      new RequestBatchIterator("A", Seq.empty[Row].iterator, emptyRequestObjectSerializer,
        emptyResponseObjectsDeserializer, true, -1, 1)
    }
    intercept[IllegalArgumentException]{
      new RequestBatchIterator("A", Seq.empty[Row].iterator, emptyRequestObjectSerializer,
        emptyResponseObjectsDeserializer, true, 1, -1)
    }
    intercept[IllegalArgumentException]{
      new RequestBatchIterator("A", Seq.empty[Row].iterator, emptyRequestObjectSerializer,
        emptyResponseObjectsDeserializer, true, -1, -1)
    }
  }

  class TestRequestObjectSerializer extends RequestRowSerializer {
    override def serializeRow(obj: Row): Array[Byte] = {
      val b = ByteBuffer.allocate(4)
      b.putInt(obj.getInt(0))
      val rv = b.array()
      rv
    }
    override val contentType: String = "None"
  }

  class TestResponseObjectsDeserializer extends ResponseRowDeserializer {
    override def deserializeResponse(inputData: Array[Byte]): Iterator[Row] = {
      var ints = Seq.empty[Row]
      val bytes = new Array[Byte](4)
      for (index <- Range(0, inputData.length, 4)) {
        Array.copy(inputData, index, bytes, 0, 4)
        val nextInt = ByteBuffer.wrap(bytes).getInt
        ints = ints :+ Row(nextInt)
      }
      ints.iterator
    }
    override val schema = new StructType(Array(StructField("int", IntegerType)))
    override val accepts: String = "None"
  }

  /**
    * Tests batching of records and result row values for the RequestBatchIterator
    *
    * Generates an input sequence of records containing numSourceRecords. Each record is an
    * integer from 0 to numSourceRecords - 1. This is then batched for transformation using the
    * RequestBatchIterator with the specified maxBatchSizeInBytes, maxBatchSizeInRecords and
    * prependInputRows.
    *
    * The TestSageMakerTransformer is used to transform data. TestRequestObjectSerializer and
    * TestResponseObjectDeserializer are used to handle the serialization and deserialization to and
    * from the TestSageMakerTransformer.
    */
  def runBasicTest(numSourceRecords : Int, maxBatchSizeInBytes : Int,
                   maxBatchSizeInRecords : Int, prependResultRows : Boolean = false) : Unit = {
    val sourceIterator = getRowIterator(numSourceRecords)

    val requestBatchIterator = new RequestBatchIterator(
      "an endpoint name",
      sourceIterator,
      new TestRequestObjectSerializer(),
      new TestResponseObjectsDeserializer(),
      maxBatchSizeInBytes = maxBatchSizeInBytes,
      maxBatchSizeInRecords = maxBatchSizeInRecords,
      prependResultRows = prependResultRows)

    var baseValue = 0
    while (requestBatchIterator.hasNext) {
      val nextValue = requestBatchIterator.next()
      if (prependResultRows) {
        assert(nextValue.getInt(0) == baseValue)
        assert(nextValue.getInt(1) == baseValue * 2)
        assert(nextValue.schema.fields.length == 2)
      }
      else {
        assert(nextValue.getInt(0) == baseValue * 2)
      }
      baseValue += 1
    }
    assert(baseValue == numSourceRecords)
  }

  it should "process multiple batches of data with single records in each batch by batch byte " +
    "size" in {
    runBasicTest(33, 4, 1000)
    runBasicTest(33, 4, 1000, prependResultRows = true)
  }

  it should "process multiple batches of data with multiple records in each batch by batch byte " +
    "size" in {
    runBasicTest(33, 8, 1000)
  }

  it should "process multiple batches of data with irregular batch byte size in each batch" in {
    runBasicTest(33, 9, 1000) // Each record occupies 4 bytes, but we have a batch size of 9
    runBasicTest(33, 5, 1000) // Each record occupies 4 bytes, but we have a batch size of 5
  }

  it should "fail if batch size by bytes is too small" in {
    intercept[IllegalArgumentException] {runBasicTest(33, 3, 1000)}
  }

  it should "process multiple batches of data with single records in each batch by record size" in {
    runBasicTest(33, 100, 1)
  }

  it should "process multiple batches of data with multiple records in each batch by record " +
    "size" in {
    runBasicTest(33, 100, 4)
  }

  it should "throw NoSuchElementException when no elements left after single batch" in {
    val sourceIterator = getRowIterator(4)

    val requestBatchIterator = new RequestBatchIterator(
      "an endpoint name",
      sourceIterator,
      new TestRequestObjectSerializer(),
      new TestResponseObjectsDeserializer(),
      maxBatchSizeInRecords = 5)

    Seq.range(0, 4).foreach(_ => requestBatchIterator.next())
    intercept[NoSuchElementException]{requestBatchIterator.next()}
  }

  it should "throw NoSuchElementException when no elements left after multiple batches" in {
    val sourceIterator = getRowIterator(8)

    val requestBatchIterator = new RequestBatchIterator(
      "an endpoint name",
      sourceIterator,
      new TestRequestObjectSerializer(),
      new TestResponseObjectsDeserializer(),
      maxBatchSizeInBytes = 1000,
      maxBatchSizeInRecords = 4)

    Seq.range(0, 8).foreach(_ => requestBatchIterator.next())
    intercept[NoSuchElementException]{requestBatchIterator.next()}
  }

  it should "throw NoSuchElementException when no elements left with partially full batch" in {
    val sourceIterator = getRowIterator(9)

    val requestBatchIterator = new RequestBatchIterator(
      "an endpoint name",
      sourceIterator,
      new TestRequestObjectSerializer(),
      new TestResponseObjectsDeserializer(),
      maxBatchSizeInBytes = 1000,
      maxBatchSizeInRecords = 4)

    Seq.range(0, 9).foreach(_ => requestBatchIterator.next())
    intercept[NoSuchElementException]{requestBatchIterator.next()}
  }

  private def getRowIterator(numSourceRecords: Int) : Iterator[Row] = {
    Seq.range(0, numSourceRecords).map(a => new GenericRowWithSchema(
        Array(a), StructType(Array(StructField("a", IntegerType))))).iterator
  }
}
