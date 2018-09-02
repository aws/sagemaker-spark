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

import org.apache.spark.ml.linalg.{DenseMatrix, DenseVector, Matrix, SparseMatrix, SparseVector, Vector}
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
      val idx = row.fieldIndex(featuresFieldName)
      val target = row.get(idx) match {
        case v : Vector =>
          setFeatures(protobufBuilder, v)
        case m : Matrix =>
          setFeatures(protobufBuilder, m)
      }
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

  /*
   *  DenseVector is encoded as
   *      Record.values([value_0, value_1, value_2, ...])
   *
   *  SparseVector is encoded as:
   *      Record.values([value_0, value_1, value_2, ...])
   *      Record.keys([value_0_idx, value_1_idx, value_2_idx])
   *      Record.shape(length)
   *  For instance the SparseVector[0, 1, 0, 2] is encoded as:
   *      Record.values([1, 2])
   *      Record.keys([1, 3])
   *      Record.shape(4)
   */
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

  /*
   *  DenseMatrix is stored in column major as
   *      Record.values([value_0, value_1, value_2, ...])
   *      Record.shape([rows, cols])
   *
   *  SparseMatrix is stored as Compressed Sparse Column (CSC)
   *  (See Spark SparseMatrix documentation)
   *      Record.values([value_0, value_1, value_2, ...])
   *      Record.shape([rows, cols])
   *      Record.keys([index_0, index_1, index_2, ...]) where
   *          row_i = floor(index_i / cols)
   *          col_i = index_i % cols
   *
   *  For instance the SparseMatrix[ 0.0  1.0  4.0  0.0
   *                                 0.0  2.0  5.0  6.0
   *                                 0.0  3.0  0.0  0.0 ] is encoded as:
   *     Record.values[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
   *     Record.shape[3, 4]
   *     Record.keys[1, 5, 9, 2, 6, 7]
   *
   *  For SparseMatrix, since each key univocally identifyes row/col there is
   *  no assumed order for the keys or values.
   */
  private def setFeatures(protobufBuilder: Record.Builder,
                          matrix: Matrix): Record.Builder = {
    val featuresTensorBuilder = Value.newBuilder().getFloat32TensorBuilder()

    // Matrix shape must be recorded for both dense/sparse matrices
    featuresTensorBuilder.addShape(matrix.numRows)
    featuresTensorBuilder.addShape(matrix.numCols)

    val featuresTensor = matrix match {
      case m: DenseMatrix =>
        if (!m.isTransposed) {
          // Convert to Row major order
          for (row <- (0 to m.numRows - 1)) {
            for (col <- (0 to m.numCols - 1)) {
              featuresTensorBuilder.addValues(m(row, col).toFloat)
            }
          }
        } else {
          // When transposed it is already in Row major order
          for (value <- m.values) {
            featuresTensorBuilder.addValues(value.toFloat)
          }
        }

        featuresTensorBuilder.build()

      case m: SparseMatrix =>
        // Construct the index for each value so that
        // row = floor(index / cols)
        // col = index % cols
        var rowIdx = 0
        var colIdx = 0
        for (colStart <- m.colPtrs.slice(1, m.colPtrs.size)) {
          while (rowIdx < colStart) {
            if (m.isTransposed) {
              // When transposed, rowIndices behave are colIndices, and colPtrs and rowPtrs
              // and rowIdx, colIdx are swapped
              featuresTensorBuilder.addKeys((colIdx * m.numCols) + m.rowIndices(rowIdx))
            } else {
              featuresTensorBuilder.addKeys((m.rowIndices(rowIdx) * m.numCols) + colIdx)
            }
            rowIdx += 1
          }
          colIdx += 1
        }

        for (value <- m.values) {
          featuresTensorBuilder.addValues(value.toFloat)
        }

        featuresTensorBuilder.build()
    }
    val featuresValue = Value.newBuilder().setFloat32Tensor(featuresTensor).build
    val mapEntry = MapEntry.newBuilder().setKey(ValuesIdentifierString)
      .setValue(featuresValue).build
    protobufBuilder.addFeatures(mapEntry)
  }

}
