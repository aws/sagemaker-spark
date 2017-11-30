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

import org.apache.spark.sql.Row
import org.apache.spark.sql.types.StructType

import com.amazonaws.services.sagemaker.sparksdk.protobuf.ProtobufConverter
import com.amazonaws.services.sagemaker.sparksdk.transformation.{ContentTypes, RequestRowSerializer}

/**
  * A [[RequestRowSerializer]] for converting labeled rows to SageMaker Protobuf-in-recordio
  * request data.
  *
  * @param schema The schema of Rows being serialized. This parameter is Optional as the schema
  *               may not be known when this serializer is constructed.
  * @param featuresColumnName the name of the features column
  */
class ProtobufRequestRowSerializer(val schema : Option[StructType] = None,
                                   val featuresColumnName : String = "features")
  extends RequestRowSerializer {

  if (schema.isDefined) {
    setSchema(schema.get)
  }

  /**
    * Serializes an object to an Array of Record-IO encoded Protobuf bytes for transformation
    * by a SageMaker endpoint
    *
    * @param row The row to serialize
    * @return An Array[Byte]
    */
  override def serializeRow(row: Row): Array[Byte] = {
    val protobuf = ProtobufConverter.rowToProtobuf(row, featuresColumnName)
    ProtobufConverter.byteArrayToRecordIOEncodedByteArray(protobuf.toByteArray)
  }

  /**
    * Returns [[com.amazonaws.services.sagemaker.sparksdk.transformation.ContentTypes.PROTOBUF]]
    */
  override val contentType: String = ContentTypes.PROTOBUF

  /**
    * Validates that the specified schema contains a Vector column with name featuresColumnName.
    *
    * @throws java.lang.IllegalArgumentException if the specified schema is invalid
    */
  override def validateSchema(schema : StructType): Unit = {
    SchemaValidators.unlabeledSchemaValidator(schema, featuresColumnName)
  }

}
