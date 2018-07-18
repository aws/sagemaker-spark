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

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.collection.mutable.ListBuffer

import aialgorithms.proto2.RecordProto2.{Float32Tensor, MapEntry, Record, Value}

import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.sql.Row
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.types._

import com.amazonaws.services.sagemaker.sparksdk.protobuf.ProtobufConverter
import com.amazonaws.services.sagemaker.sparksdk.transformation.{ContentTypes, ResponseRowDeserializer}

/**
  * A [[ResponseRowDeserializer]] for converting SageMaker Protobuf-in-recordio response data
  * to Spark rows.
  * @param schema The schema of rows in the response.
  * @param protobufKeys The keys for each output in the Protobuf response.
  */
class ProtobufResponseRowDeserializer(override val schema: StructType,
                                      val protobufKeys : Option[Seq[String]] = None)
  extends ResponseRowDeserializer {

  /**
    * Deserialize a SageMaker response containing a byte array of record-IO encoded SageMaker
    * Protobuf records.
    *
    * @param responseData The Array[Byte] containing the Protobuf-in-recordIO response
    * @return An Iterator over deserialized Rows
    */
  override def deserializeResponse(responseData: Array[Byte]): Iterator[Row] = {
    val recordIterator = ProtobufConverter.recordIOByteArrayToProtobufs(responseData)
    val rowList = ListBuffer[Row]()
    while (recordIterator.hasNext) {
      rowList += protobufToRow(recordIterator.next)
    }
    rowList.toIterator
  }

  private def protobufToRow(record: Record) : Row = {
    val rowValues = ListBuffer[Any]()
    val labelList = record.getLabelList
    for (fieldName <- protobufKeys.getOrElse[Seq[String]](schema.fieldNames)) {
      val value = getValueFromLabelList(fieldName, labelList.asScala)
      val dataValue = getValueFromTensor(value.getFloat32Tensor)
      rowValues += dataValue
    }
    if (schema.fields.length > rowValues.toArray.length) {
      throw new IllegalArgumentException(s"Expected ${schema.fields.length} entries to fit" +
        s" schema ${schema}, received ${rowValues.toArray.length} entries: ${rowValues.toList}")
    }
    new GenericRowWithSchema(rowValues.toArray, schema)
  }

  private def getValueFromLabelList(fieldName: String,
                                    labelList: mutable.Buffer[MapEntry]): Value = {
    val mapEntryList = labelList.filter(_.getKey == fieldName)
    require(mapEntryList.nonEmpty, s"Couldn't find field $fieldName in $labelList")
    mapEntryList.head.getValue
  }

  private def getValueFromTensor(valuesTensor: Float32Tensor) : Any = {
    val valuesCount = valuesTensor.getValuesCount
    require(valuesCount > 0, "Can't get value from deserialized tensor: values list is empty.")

    if (valuesCount == 1) {
      // Get value as a scalar
      valuesTensor.getValues(0).toDouble
    } else {
      // Get value as a Vector
      val keyCount = valuesTensor.getKeysCount
      val values = asScalaBufferConverter(valuesTensor.getValuesList)
        .asScala.toArray.map(_.toDouble)
        if (keyCount > 0) {
        // Tensor is sparsely encoded. We can only represent sparsely-encoded vectors
        // (not higher-order tensors).
        require(valuesTensor.getShapeCount == 1,
          "Cannot deserialize tensor to vector. Shape list has more than one value.")
        val indices = asScalaBufferConverter(valuesTensor.getKeysList).asScala.toArray.map(_.toInt)
        return new SparseVector(valuesTensor.getShape(0).toInt, indices, values)
      } else {
        // Vector is densely encoded.
        return new DenseVector(values)
      }
    }
  }

  /**
    * Returns [[com.amazonaws.services.sagemaker.sparksdk.transformation.ContentTypes.PROTOBUF]]
    */
  override val accepts: String = ContentTypes.PROTOBUF
}

/**
  * Deserializes a Protobuf response from the KMeans model image into Rows in a Spark Dataframe
  *
  * @param distanceToClusterColumnName name of the column of doubles indicating the distance to
  *   the nearest cluster from the input record.
  * @param closestClusterColumnName name of the column of doubles indicating the label of the
  *   closest cluster for the input record.
  */
class KMeansProtobufResponseRowDeserializer
(val distanceToClusterColumnName: String = "distance_to_cluster",
 val closestClusterColumnName: String = "closest_cluster")
  extends ProtobufResponseRowDeserializer(schema =
    StructType(Array(StructField(distanceToClusterColumnName, DoubleType),
      StructField(closestClusterColumnName, DoubleType))),
    protobufKeys = Some(Seq("distance_to_cluster", "closest_cluster")))

/**
  * Deserializes a Protobuf response from the PCA model image to a Vector of Doubles containing
  *   the projection of the input vector.
  *
  * @param projectionColumnName name of the column holding Vectors of Doubles representing the
  *   projected vectors
  */
class PCAProtobufResponseRowDeserializer
(val projectionColumnName : String = "projection")
  extends ProtobufResponseRowDeserializer(schema =
    StructType(Array(StructField(projectionColumnName, VectorType))),
    protobufKeys = Some(Seq("projection")))

/**
  * Deserializes a Protobuf response from the LinearLearner model image with predictorType
  *   "binary_classifier" into Rows in a Spark Dataframe
  * @param scoreColumnName name of the column of Doubles indicating the output score for the record.
  * @param predictedLabelColumnName name of the column of Doubles indicating the predicted label
  *   for the record.
  */
class LinearLearnerBinaryClassifierProtobufResponseRowDeserializer
(val scoreColumnName: String = "score",
 val predictedLabelColumnName: String = "predicted_label")
  extends ProtobufResponseRowDeserializer(schema =
    StructType(Array(StructField(scoreColumnName, DoubleType),
      StructField(predictedLabelColumnName, DoubleType))),
    protobufKeys = Some(Seq("score", "predicted_label")))

/**
  * Deserializes a Protobuf response from the LinearLearner model image with predictorType
  *   "multiclass_classifier" into Rows in a Spark Dataframe
  * @param scoreColumnName name of the column of Vectors indicating the output score for the record.
  * @param predictedLabelColumnName name of the column of Doubles indicating the predicted label
  *   for the record.
  */
class LinearLearnerMultiClassClassifierProtobufResponseRowDeserializer
(val scoreColumnName: String = "score",
 val predictedLabelColumnName: String = "predicted_label")
  extends ProtobufResponseRowDeserializer(schema =
    StructType(Array(StructField(scoreColumnName, VectorType),
      StructField(predictedLabelColumnName, DoubleType))),
    protobufKeys = Some(Seq("score", "predicted_label")))

/**
  * Deserializes a Protobuf response from the LinearLearner model image with predictorType
  * "regressor" into Rows in a Spark Dataframe
  *
  * @param scoreColumnName name of the column of Doubles indicating the output score for the record.
  */
class LinearLearnerRegressorProtobufResponseRowDeserializer
(val scoreColumnName: String = "score")
  extends ProtobufResponseRowDeserializer(schema =
    StructType(Array(StructField(scoreColumnName, DoubleType))),
    protobufKeys = Some(Seq("score")))

/**
  * Deserializes a Protobuf response from the FactorizationMachines model image with predictorType
  *   "binary_classifier" into Rows in a Spark Dataframe
  * @param scoreColumnName name of the column of Doubles indicating the output score for the record.
  * @param predictedLabelColumnName name of the column of Doubles indicating the predicted label
  *   for the record.
  */
class FactorizationMachinesBinaryClassifierDeserializer
(val scoreColumnName: String = "score",
 val predictedLabelColumnName: String = "predicted_label")
  extends ProtobufResponseRowDeserializer(schema =
    StructType(Array(StructField(scoreColumnName, DoubleType),
      StructField(predictedLabelColumnName, DoubleType))),
    protobufKeys = Some(Seq("score", "predicted_label")))

/**
  * Deserializes a Protobuf response from the FactorizationMachines model image with predictorType
  * "regressor" into Rows in a Spark Dataframe
  *
  * @param scoreColumnName name of the column of Doubles indicating the output score for the record.
  */
class FactorizationMachinesRegressorDeserializer
(val scoreColumnName: String = "score")
  extends ProtobufResponseRowDeserializer(schema =
    StructType(Array(StructField(scoreColumnName, DoubleType))),
    protobufKeys = Some(Seq("score")))

/**
  * Deserializes a Protobuf response from the LDA model image to a Vector of Doubles
  *   representing the topic mixture for the document represented by the input vector.
  *
  * @param topicMixtureColumnName name of the column holding Vectors of Doubles representing the
  *   topic mixtures for the documents
  */
class LDAProtobufResponseRowDeserializer
(val topicMixtureColumnName : String = "topic_mixture")
  extends ProtobufResponseRowDeserializer(schema =
    StructType(Array(StructField(topicMixtureColumnName, VectorType))),
    protobufKeys = Some(Seq("topic_mixture")))
