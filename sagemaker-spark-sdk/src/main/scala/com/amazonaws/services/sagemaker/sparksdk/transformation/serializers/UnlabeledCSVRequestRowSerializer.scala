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

package com.amazonaws.services.sagemaker.sparksdk.transformation.serializers

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.StructType

import com.amazonaws.services.sagemaker.sparksdk.transformation.{ContentTypes, RequestRowSerializer}

/**
  * Serializes according to the current implementation of the scoring service:
  */
class UnlabeledCSVRequestRowSerializer(val schema : Option[StructType] = None,
                                       val featuresColumnName : String = "features")
  extends RequestRowSerializer with java.io.Serializable {

  private var featuresFieldIndex : Int = _

  if(schema.isDefined) {
    setSchema(schema.get)
  }

  /**
    * Serializes an object to an Array of bytes for transformation by a SageMaker endpoint
    *
    * @param row The row to serialize
    * @return An Array[Byte]
    */
  override def serializeRow(row: Row): Array[Byte] = {
    val features = row.getAs[Vector](featuresFieldIndex)
    serializeVector(features)
  }

  private def serializeVector(features: Vector): Array[Byte] = {
    (features.toDense.values.mkString(",") + "\n").getBytes
  }
  /**
    * Returns [[com.amazonaws.services.sagemaker.sparksdk.transformation.ContentTypes.CSV]]
    */
  override val contentType: String = ContentTypes.CSV

  /**
    * Validates that the specified schema contains a Double column with name labelColumnName and
    * a Vector column with name featuresColumnName.
    *
    * @throws java.lang.IllegalArgumentException if the specified schema is invalid
    */
  override def validateSchema(schema : StructType): Unit = {
    SchemaValidators.unlabeledSchemaValidator(schema, featuresColumnName)
  }

  /**
    * @inheritdoc
    *
    * This method must be invoked before calling serializeRow if no schema was set when this
    * RequestRowSerializer was constructed.
    */
  override def setSchema(schema : StructType): Unit = {
    super.setSchema(schema)
    this.featuresFieldIndex = schema.fieldIndex(featuresColumnName)
  }

}
