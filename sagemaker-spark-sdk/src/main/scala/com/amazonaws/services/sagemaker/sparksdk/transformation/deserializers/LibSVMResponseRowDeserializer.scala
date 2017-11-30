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

package com.amazonaws.services.sagemaker.sparksdk.transformation.deserializers

import org.apache.spark.ml.linalg.{SparseVector, SQLDataTypes}
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}

import com.amazonaws.services.sagemaker.sparksdk.transformation.{ContentTypes, ResponseRowDeserializer}

/**
  * A [[ResponseRowDeserializer]] for converting LibSVM response data to labeled vectors.
  *
  * @param dim The vector dimension
  * @param labelColumnName The name of the label column
  * @param featuresColumnName The name of the features column
  */
class LibSVMResponseRowDeserializer (val dim : Int,
                                     val labelColumnName : String = "label",
                                     val featuresColumnName : String = "features")
  extends ResponseRowDeserializer {

  if (dim < 0) {
    throw new IllegalArgumentException("Vector dimension must not be negative")
  }

  /**
    * Deserialize a SageMaker response to an a series of objects.
    *
    * @param responseData The Array[Byte] containing the SageMaker response
    * @return An Iterator over deserialized response objects
    */
  override def deserializeResponse(responseData: Array[Byte]): Iterator[Row] = {
    new String(responseData).split("\n").map(parseLibSVMRow).iterator
  }

  /**
    * Returns [[com.amazonaws.services.sagemaker.sparksdk.transformation.ContentTypes.TEXT_LIBSVM]]
    */
  override val accepts: String = ContentTypes.TEXT_LIBSVM

  private def parseLibSVMRow(record: String): Row = {
    val items = record.split(' ')
    val label = items.head.toDouble
    val (indices, values) = items.tail.filter(_.nonEmpty).map { item =>
      val entry = item.split(':')
      val index = entry(0).toInt - 1
      val value = entry(1).toDouble
      (index, value)
    }.unzip
    Row(label, new SparseVector(dim, indices.toArray, values.toArray))
  }

  override val schema: StructType = StructType(
    Array(
      StructField(labelColumnName, DoubleType, nullable = false),
      StructField(featuresColumnName, SQLDataTypes.VectorType, nullable = false)))
}
