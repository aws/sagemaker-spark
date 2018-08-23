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

package com.amazonaws.services.sagemaker.sparksdk.transformation.util

import java.nio.ByteBuffer

import scala.collection.mutable
import scala.collection.mutable.ListBuffer

import com.amazonaws.ClientConfiguration
import com.amazonaws.services.sagemakerruntime.{AmazonSageMakerRuntime, AmazonSageMakerRuntimeClientBuilder}
import com.amazonaws.services.sagemakerruntime.model.InvokeEndpointRequest
import com.amazonaws.util.BinaryUtils

import org.apache.spark.sql.Row
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.types.StructType

import com.amazonaws.services.sagemaker.sparksdk.transformation.{RequestRowSerializer, ResponseRowDeserializer}

object RequestBatchIterator {
  var sagemakerRuntime : AmazonSageMakerRuntime =
    AmazonSageMakerRuntimeClientBuilder
      .standard()
      .withClientConfiguration(new ClientConfiguration().withSocketTimeout(80 * 1000))
      .build()
}

/**
  * Iterates over SageMaker transformed Rows by transforming input [[Row]]s to output Rows using
  * a SageMaker Endpoint.
  *
  * SageMaker Transformation is done in batch. Input records are read from sourceIterator and
  * serialized to an Array[Byte] using a [[RequestRowSerializer]]. These byte arrays are
  * concatenated into a batch request containing at most maxBatchSizeInRecords records and at
  * most maxBatchSizeInBytes number of bytes. Transformation is then performed by invoking
  * [[AmazonSageMakerRuntime#invokeEndpoint]] on an [[AmazonSageMakerRuntime]], using the Byte
  * Array as the request body.
  *
  * The transformation response as an Array of Bytes is converted to a series of Rows
  * by a [[ResponseRowDeserializer]]
  *
  * The SageMaker transformation uses a content type retrieved from
  * [[RequestRowSerializer#contentType]] and an accepts from [[ResponseRowDeserializer#accepts]].
  * The SageMaker transformation response is converted to an Iterator of Rows by
  * responseRowDeserializer.
  *
  * @param endpointName The name of the SageMaker endpoint to invoke
  * @param sourceIterator Input Rows are read from this iterator
  * @param requestRowSerializer Serializes each Row to a Byte Array for SageMaker transformation
  * @param responseRowDeserializer Deserializes each SageMaker Endpoint transformation response to a
  *                                series of Rows
  * @param prependResultRows Whether output Rows returned by this Iterator should be prepended with
  *                          the input Rows read from sourceIter
  * @param maxBatchSizeInRecords The maximum number of records to send in each batch to the
  *                              SageMaker endpoint
  * @param maxBatchSizeInBytes The maximum byte size to send in each batch to the SageMaker endpoint
  */
 private[sparksdk] class RequestBatchIterator (
                                              val endpointName : String,
                                              val sourceIterator : Iterator[Row],
                                              val requestRowSerializer: RequestRowSerializer,
                                              val responseRowDeserializer: ResponseRowDeserializer,
                                              val prependResultRows : Boolean = true,
                                              val maxBatchSizeInRecords : Int = Int.MaxValue,
                                              val maxBatchSizeInBytes : Int = 5242880)
  extends Iterator[Row] {

  if (maxBatchSizeInRecords < 1) {
    throw new IllegalArgumentException("maxBatchSizeInRecords < 1")
  }

  if (maxBatchSizeInBytes < 1) {
    throw new IllegalArgumentException("maxBatchSizeInBytes < 1")
  }

  /**
    * Stores transformed records to return from this iterator.
    */
  private var currentResultsIterator = ListBuffer.empty[Row].iterator

  /**
    * A convenience wrapper for sourceIterator and requestObjectSerializer.
    * Converts the sourceIterator into an Iterator[Array[Byte]].
    */
  private val requestObjectIterator = new RequestRowIterator(
    sourceIterator,
    maxBatchSizeInBytes,
    requestRowSerializer)

  override def hasNext : Boolean = {
    // Iteratively construct currentResultsIterator from SageMaker requests until
    // a currentResultsIterator exists with at least one record to return.
    while (!currentResultsIterator.hasNext) {
      if (!requestObjectIterator.hasNext) { // Return false if no input records left to transform
        return false
      }
      // Construct a SageMaker request as an array of bytes by taking objects from the
      // sourceIter and serializing hem with requestObjectSerializer
      var batchSizeInRecords = 0
      val batchBody = ByteBuffer.allocate(maxBatchSizeInBytes)
      val batchInputRows = new mutable.ListBuffer[Row]()
      while(requestObjectIterator.hasNext &&
        requestObjectIterator.nextValueLength + batchBody.position <= maxBatchSizeInBytes &&
        batchSizeInRecords <= maxBatchSizeInRecords) {
        val (nextRow, nextSerializedRow) = requestObjectIterator.next()
        if(prependResultRows) {
          batchInputRows += nextRow
        }
        batchBody.put(nextSerializedRow)
        batchSizeInRecords += 1
      }
      batchBody.flip()
      val invokeEndpointRequest = new InvokeEndpointRequest()
        .withEndpointName(endpointName)
        .withContentType(requestRowSerializer.contentType)
        .withAccept(responseRowDeserializer.accepts)
        .withBody(batchBody)
      val endpointResponse = RequestBatchIterator.sagemakerRuntime
        .invokeEndpoint(invokeEndpointRequest).getBody

      val resultsIterator = responseRowDeserializer
          .deserializeResponse(BinaryUtils.copyBytesFrom(endpointResponse))

      // If result rows should have input rows prepended, then wrap the resultsIterator from
      // the deserializer with an iterator that does the prepending
      if (prependResultRows) {
        currentResultsIterator = new Iterator[Row] {
          val batchInputRowsIterator = batchInputRows.iterator.buffered
          val row = batchInputRowsIterator.head
          val schema = row.schema
          assert(schema != null,
            "Input rows must have a schema when prepending input rows to result rows")
          val newSchema = StructType(batchInputRowsIterator.head.schema
            ++ responseRowDeserializer.schema)
          override def hasNext: Boolean = resultsIterator.hasNext
          override def next(): Row = {
            new GenericRowWithSchema(
              (batchInputRowsIterator.next.toSeq ++ resultsIterator.next.toSeq).toArray,
              newSchema)
          }
        }
      } else {
        currentResultsIterator = resultsIterator
      }
    }
    true
  }

  override def next() : Row = {
    if (currentResultsIterator.hasNext) {
      return currentResultsIterator.next
    }
    if (!hasNext) {
      throw new NoSuchElementException
    }
    currentResultsIterator.next
  }
}

/**
  * Translates an Iterator[Row] to an Iterator[(Row, Array[Byte])] via a RequestObjectSerializer[T].
  */
private class RequestRowIterator (val input: Iterator[Row],
                                  val maxBatchSizeInBytes : Int,
                                  val requestObjectSerializer: RequestRowSerializer)
  extends Iterator[(Row, Array[Byte])] {

  // Facilitate peeking by eagerly reading the first value from sourceIterator, if it exists
  var nextValue : Option[(Row, Array[Byte])] = Option.empty

  if (input.hasNext) {
    nextValue = Option(serializeRow(input.next))
  }

  /**
    * Returns the length in bytes of the next serialized row or 0 if there is no next row
    */
  def nextValueLength : Int = {
    nextValue map {case (a, b) => b.length} getOrElse 0
  }

  override def hasNext: Boolean = {
    nextValue.nonEmpty
  }

  override def next(): (Row, Array[Byte]) = {
    val returnVal = nextValue.get
    if(input.hasNext) {
      nextValue = Option(serializeRow(input.next))
    } else {
      nextValue = Option.empty
    }
    returnVal
  }

  private def serializeRow(obj : Row): (Row, Array[Byte]) = {
    val serialized = requestObjectSerializer.serializeRow(obj)
    if (serialized.length > maxBatchSizeInBytes) {
      throw new IllegalArgumentException("Object serialized to byte array of length: "
        + serialized.length + " which is above max batch size in bytes of "
        + maxBatchSizeInBytes + ". Object: " + obj)
    }
    (obj, serialized)
  }
}

object RequestBatchIteratorFactory extends Serializable {

  /**
    * Creates [[RequestBatchIterator]] for SageMakerModel.
    *
    * @param endpointName The name of an endpoint that is current in service.
    * @param ser Serializes a Row to an Array of Bytes.
    * @param de Deserializes an Array of Bytes to a series of Rows.
    * @param prependResultRows Whether the transformation result should include the input Rows.
    * @return An iterator.
    */
  def createRequestBatchIterator(endpointName : String,
                                 ser : RequestRowSerializer,
                                 de : ResponseRowDeserializer,
                                 prependResultRows : Boolean):
  Iterator[Row] => RequestBatchIterator = (input : Iterator[Row]) => {
    new RequestBatchIterator(
      endpointName,
      input,
      ser,
      de,
      prependResultRows)
  }
}
