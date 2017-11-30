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

package com.amazonaws.services.sagemaker.sparksdk.transformation

import org.apache.spark.sql.Row
import org.apache.spark.sql.types.StructType

/**
  * Serializes a [[Row]] to a byte array with content type [[RequestRowSerializer#contentType]].
  *
  * Rows are serialized with [[RequestRowSerializer#serializeRow]] to an Array of Bytes.
  * Implementations may require that Rows being serialized conform to a specific schema. Row
  * schemas can be validated by [[RequestRowSerializer#validateSchema]]. The schema for a batch
  * of Row of objects can be set via [[RequestRowSerializer#setSchema]].
  *
  * @see [[com.amazonaws.services.sagemaker.sparksdk.SageMakerModel]]
  */
trait RequestRowSerializer extends Serializable {

  /**
    * The schema for Row objects being serialized by [[RequestRowSerializer#serializeRow]]
    */
  protected var rowSchema : StructType = _

  /**
    * Serializes a [[Row]] to an Array of Bytes, conforming to a content-type of
    * [[RequestRowSerializer#contentType]]
    *
    * @param row The row to serialize
    * @return An Array[Byte]
    */
  def serializeRow(row : Row) : Array[Byte]

  /**
    * The SageMaker transformation content-type
    */
  val contentType : String

  /**
    * Validates that [[Row]]s with the specified schema can be serialized by this
    * [[RequestRowSerializer]].
    *
    * Validation errors are reported by throwing an [[IllegalArgumentException]]
    *
    * @throws IllegalArgumentException if Rows with the specified schema cannot be serialized
    *                                  by this [[RequestRowSerializer]]
    */
  def validateSchema(schema : StructType): Unit = {

  }

  /**
    * Sets the rowSchema for this RequestRowSerializer. Invokes validateSchema on the
    * specified schema.
    */
  def setSchema(schema : StructType): Unit = {
    validateSchema(schema)
    this.rowSchema = schema
  }
}
