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

package com.amazonaws.services.sagemaker.sparksdk.protobuf

import java.nio.{ByteBuffer, ByteOrder}

import scala.collection.mutable

import aialgorithms.proto2.RecordProto2
import aialgorithms.proto2.RecordProto2.{MapEntry, Record, Value}
import aialgorithms.proto2.RecordProto2.Record.Builder

import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector}
import org.apache.spark.sql.Row

/**
  * Utility functions that convert to and from the Amazon Record protobuf data format and encode
  * Records in recordIO.
  *
  * @see [[https://aws.amazon.com/sagemaker/latest/dg/cdf-training.html/]] for more information on
  *     the Amazon Record data format
  * @see [[https://mxnet.incubator.apache.org/architecture/note_data_loading.html]] for more
  *      information on recordIO
  */
object ProtobufConverter {

  /** Certain first-party algorithms (like K-Means, PCA, and LinearLearner) on Amazon SageMaker
    * expect each Amazon Record protobuf message to contain a "features" map with key "values"
    * that encodes the example's features vector.
    */
  val ValuesIdentifierString = "values"

  /**
    * Converts a Row to a SageMaker Protobuf record for training or inference.
    *
    * @param row a Row with a column of Doubles for labels and a Vector for features
    * @param featuresFieldName index in the row corresponding to the features column
    * @param labelFieldName index in the row corresponding to the label column
    * @return a SageMaker Protobuf record representing the Row
    */
  def rowToProtobuf(row: Row, featuresFieldName: String,
                    labelFieldName: Option[String] = Option.empty): Record = {
    require(row.schema != null, "Row schema is null for row " + row)
    val protobufBuilder : Builder = Record.newBuilder()

    if (labelFieldName.nonEmpty) {
      val hasLabelColumn = row.schema.fieldNames.contains(labelFieldName.get)
      if (hasLabelColumn) {
        setLabel(protobufBuilder, row.getAs[Double](labelFieldName.get))
      }
    }
    val hasFeaturesColumn = row.schema.fieldNames.contains(featuresFieldName)

    if (hasFeaturesColumn) {
      val vector = row.getAs[Vector](featuresFieldName)
      setFeatures(protobufBuilder, vector)
    } else if (!hasFeaturesColumn) {
      throw new IllegalArgumentException(s"Need a features column with a " +
        s"Vector of doubles named $featuresFieldName to convert row to protobuf")
    }
    protobufBuilder.build
  }

  /**
    * Converts a recordIO encoded byte array of Protobuf records to an iterator of Protobuf Records
    *
    * @see [[https://mxnet.incubator.apache.org/architecture/note_data_loading.html]] for more
    *     information on recordIO
    *
    * @param byteArray recordIO encoded byte array to convert to Protobuf Record
    * @return An Iterator over Protobuf Records
    */
  def recordIOByteArrayToProtobufs(byteArray: Array[Byte]) : Iterator[Record] = {
    val recordList = new mutable.MutableList[Record]()
    val buffer = ByteBuffer.wrap(byteArray)
    buffer.order(ByteOrder.LITTLE_ENDIAN)

    var index = 0
    while (buffer.hasRemaining) {
      validateMagicNumber(buffer.getInt)

      val recordBytes = new Array[Byte](buffer.getInt)
      buffer.get(recordBytes, 0, recordBytes.length)

      val protobufRecord = ProtobufConverter.byteArrayToProtobuf(recordBytes)
      recordList += protobufRecord

      buffer.position(buffer.position + paddingCount(buffer.position))
    }
    recordList.iterator
  }

  private val magicNumber : Integer = 0xced7230a;
  private val magicNumberBytes : Array[Byte] = intToLittleEndianByteArray(magicNumber)

  private def intToLittleEndianByteArray(int : Integer) : Array[Byte] = {
    ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(int).array
  }

  /**
    * Encodes a byte array in recordIO.
    *
    * @see [[https://mxnet.incubator.apache.org/architecture/note_data_loading.html]] for more
    *     information on recordIO
    *
    * @param byteArray Byte array to encode in recordIO
    * @return RecordIO encoded byte array
    */
  def byteArrayToRecordIOEncodedByteArray(byteArray: Array[Byte]) :
  Array[Byte] = {
    val recordLengthBytes = intToLittleEndianByteArray(byteArray.length)
    var recordIOBytes = magicNumberBytes ++ recordLengthBytes ++ byteArray
    val paddingNeeded = paddingCount(recordIOBytes.length)
    for (i <- 1 to paddingNeeded) {
      recordIOBytes ++= Array[Byte](0)
    }
    recordIOBytes
  }

  /**
    * Returns how much padding is necessary to encode a byte array in record-IO
    *
    * @param byteCount length of the Record in bytes
    * @return how many bytes to pad
    */
  private[protobuf] def paddingCount(byteCount: Int): Int = {
    val mod = byteCount % 4
    if (mod == 0) 0 else 4 - mod
  }

  /**
    * Validates that the given integer is equal to the RecordIO magic number that delimits records.
    *
    * @param recordDelimiter magic number recognizing record format
    */
  private[protobuf] def validateMagicNumber(recordDelimiter: Integer): Unit = {
    if (recordDelimiter != magicNumber) {
      throw new RuntimeException("Incorrectly encoded byte array. " +
        "Record delimiter did not match RecordIO magic number.")
    }
  }

  /**
    * Converts a byte array to a Protobuf Record.
    *
    * @param byteArray byte array to convert to Protobuf Record
    * @return Protobuf Record from byte array
    */
  def byteArrayToProtobuf(byteArray: Array[Byte]): Record = {
    RecordProto2.Record.parseFrom(byteArray)
  }

  private def setLabel(protobufBuilder: Record.Builder, label: Double): Record.Builder = {
    val labelTensor = Value.newBuilder().getFloat32TensorBuilder.addValues(label.toFloat).build
    val labelValue = Value.newBuilder().setFloat32Tensor(labelTensor).build()
    val mapEntry = MapEntry.newBuilder().setKey(ValuesIdentifierString).setValue(labelValue).build
    protobufBuilder.addLabel(mapEntry)
  }

  private def setFeatures(protobufBuilder: Record.Builder,
                                    vector: Vector): Record.Builder = {
    val featuresTensorBuilder = Value.newBuilder().getFloat32TensorBuilder()

    val featuresTensor = vector match {
      case v: DenseVector =>
        for (value <- v.values) {
          featuresTensorBuilder.addValues(value.toFloat)
        }
        featuresTensorBuilder.build()
      case v: SparseVector =>
        featuresTensorBuilder.addShape(v.size)
        for (i <- 0 until v.indices.length) {
          featuresTensorBuilder.addKeys(v.indices(i))
          featuresTensorBuilder.addValues(v.values(i).toFloat)
        }
        featuresTensorBuilder.build()
    }
    val featuresValue = Value.newBuilder().setFloat32Tensor(featuresTensor).build
    val mapEntry = MapEntry.newBuilder().setKey(ValuesIdentifierString)
      .setValue(featuresValue).build
    protobufBuilder.addFeatures(mapEntry)
  }

}
