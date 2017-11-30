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

import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}

import com.amazonaws.services.sagemaker.sparksdk.transformation.{ContentTypes, ResponseRowDeserializer}

/**
  * A [[ResponseRowDeserializer]] for converting a comma-delimited string of predictions to
  * labeled Vectors.
  *
  * @param predictionColumnName the name of the output predictions column.
  */
class XGBoostCSVRowDeserializer(val predictionColumnName : String = "prediction")
  extends ResponseRowDeserializer {

  /**
    * Deserialize a SageMaker response to an a series of objects.
    *
    * @param responseData The Array[Byte] containing the SageMaker response
    * @return An Iterator over deserialized response objects
    */
  override def deserializeResponse(responseData: Array[Byte]): Iterator[Row] = {
    // Response is a comma-delimited string of predictions
    val stringResponse = new String(responseData)
    stringResponse.trim.split(",").map((s: String) => Row(s.toDouble)).toIterator
  }

  /**
    * Returns [[com.amazonaws.services.sagemaker.sparksdk.transformation.ContentTypes.CSV]]
    */
  override val accepts: String = ContentTypes.CSV

  override val schema: StructType = StructType(
    Array(StructField(predictionColumnName, DoubleType, nullable = false)))
}
