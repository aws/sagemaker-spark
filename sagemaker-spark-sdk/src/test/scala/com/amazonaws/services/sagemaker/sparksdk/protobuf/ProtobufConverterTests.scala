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

import scala.collection.JavaConversions._
import scala.collection.JavaConverters._

import aialgorithms.proto2.RecordProto2
import aialgorithms.proto2.RecordProto2.Record
import org.scalatest.{BeforeAndAfter, FlatSpec}
import org.scalatest.mock.MockitoSugar

import org.apache.spark.ml.linalg.{DenseMatrix, DenseVector, SparseMatrix, SparseVector}
import org.apache.spark.ml.linalg.SQLDataTypes.{MatrixType, VectorType}
import org.apache.spark.sql.Row
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}

import com.amazonaws.services.sagemaker.sparksdk.protobuf.ProtobufConverter._

class ProtobufConverterTests extends FlatSpec with MockitoSugar with BeforeAndAfter {

  val label : Double = 1.0
  val denseFeatures : DenseVector = new DenseVector((1d to 100d by 1d).toArray)
  val sparseFeatures : SparseVector = new SparseVector(100, (1 to 100 by 10).toArray,
    (1d to 100d by 10d).toArray)
  val denseMatrixFeatures : DenseMatrix = new DenseMatrix(10, 20, (1d to 200d by 1d).toArray)
  val denseMatrixFeaturesTrans : DenseMatrix =
    new DenseMatrix(10, 20, (1d to 200d by 1d).toArray, true)
  val sparseMatrixFeatures : SparseMatrix = new SparseMatrix(3, 3,
    Array(0, 2, 3, 6),
    Array(0, 2, 1, 0, 1, 2),
    Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
  val sparseMatrixFeaturesTrans : SparseMatrix = new SparseMatrix(3, 4,
    Array(0, 2, 5, 6),
    Array(1, 2, 1, 2, 3, 1),
    Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), true)
  val labelColumnName = "label"
  val featuresColumnName = "features"
  val schemaWithoutLabels : StructType = new StructType().add(StructField(
    featuresColumnName, VectorType, true))
  val schemaMatrixWithoutLabels : StructType = new StructType().add(StructField(
    featuresColumnName, MatrixType, true))
  val schemaWithLabelsAndFeatures: StructType = new StructType()
                .add(StructField(labelColumnName, DoubleType, true))
    .add(StructField(featuresColumnName, VectorType, true))
  val schemaMatrixWithLabelsAndFeatures: StructType = new StructType()
                .add(StructField(labelColumnName, DoubleType, true))
                .add(StructField(featuresColumnName, MatrixType, true))
  val denseRowWithLabels : Row = new GenericRowWithSchema(Array(label, denseFeatures),
    schemaWithLabelsAndFeatures)
  val sparseRowWithLabels : Row = new GenericRowWithSchema(Array(label, sparseFeatures),
    schemaWithLabelsAndFeatures)
  val denseMatrixRowWithLabels : Row = new GenericRowWithSchema(Array(label, denseMatrixFeatures),
    schemaMatrixWithLabelsAndFeatures)
  val denseMatrixRowWithLabelsTrans : Row =
    new GenericRowWithSchema(Array(label, denseMatrixFeaturesTrans),
    schemaMatrixWithLabelsAndFeatures)
  val sparseMatrixRowWithLabels : Row = new GenericRowWithSchema(Array(label, sparseMatrixFeatures),
    schemaMatrixWithLabelsAndFeatures)
  val sparseMatrixRowWithLabelsTrans : Row = new GenericRowWithSchema(
    Array(label, sparseMatrixFeaturesTrans), schemaMatrixWithLabelsAndFeatures)
  val denseRowWithoutLabels : Row = new GenericRowWithSchema(Array(denseFeatures),
    schemaWithoutLabels)
  val sparseRowWithoutLabels : Row = new GenericRowWithSchema(Array(sparseFeatures),
    schemaWithoutLabels)
  val denseMatrixRowWithoutLabels : Row = new GenericRowWithSchema(Array(denseMatrixFeatures),
    schemaMatrixWithoutLabels)
  val sparseMatrixRowWithoutLabels : Row = new GenericRowWithSchema(Array(sparseMatrixFeatures),
    schemaMatrixWithoutLabels)

  it should "convert a dense row without labels to protobuf" in {
    val protobufRecord = ProtobufConverter.rowToProtobuf(denseRowWithoutLabels,
      featuresColumnName, Some(labelColumnName))
    validateDense(protobufRecord, denseFeatures)
  }

  it should "convert a sparse row without labels to protobuf" in {
    val protobufRecord = ProtobufConverter.rowToProtobuf(sparseRowWithoutLabels,
      featuresColumnName, Some(labelColumnName))
    validateSparse(protobufRecord, sparseFeatures)
  }

  it should "convert a dense Matrix without labels to protobuf" in {
    val protobufRecord = ProtobufConverter.rowToProtobuf(denseMatrixRowWithoutLabels,
      featuresColumnName, Some(labelColumnName))
    validateDense(protobufRecord, denseMatrixFeatures)
  }

  it should "convert a sparse Matrix without labels to protobuf" in {
    val protobufRecord = ProtobufConverter.rowToProtobuf(sparseMatrixRowWithoutLabels,
      featuresColumnName, Some(labelColumnName))
    validateSparse(protobufRecord, sparseMatrixFeatures)
  }

  it should "convert a dense row with labels to protobuf" in {
    val protobufRecord = ProtobufConverter.rowToProtobuf(denseRowWithLabels,
      featuresColumnName, Some(labelColumnName))
    assert(label == recordToLabel(protobufRecord))
    validateDense(protobufRecord, denseFeatures)
  }

  it should "convert a sparse row with labels to protobuf" in {
    val protobufRecord = ProtobufConverter.rowToProtobuf(sparseRowWithLabels,
      featuresColumnName, Some(labelColumnName))
    assert(label == recordToLabel(protobufRecord))
    validateSparse(protobufRecord, sparseFeatures)
  }

  it should "convert a dense Matrix with labels to protobuf" in {
    val protobufRecord = ProtobufConverter.rowToProtobuf(denseMatrixRowWithLabels,
      featuresColumnName, Some(labelColumnName))
    assert(label == recordToLabel(protobufRecord))
    validateDense(protobufRecord, denseMatrixFeatures)
  }

  it should "convert a transposed dense Matrix with labels to protobuf" in {
    val protobufRecord = ProtobufConverter.rowToProtobuf(denseMatrixRowWithLabelsTrans,
      featuresColumnName, Some(labelColumnName))
    assert(label == recordToLabel(protobufRecord))
    validateDense(protobufRecord, denseMatrixFeaturesTrans)
  }

  it should "convert a transposed sparse Matrix with labels to protobuf" in {
    val protobufRecord = ProtobufConverter.rowToProtobuf(sparseMatrixRowWithLabelsTrans,
      featuresColumnName, Some(labelColumnName))
    assert(label == recordToLabel(protobufRecord))
    validateSparse(protobufRecord, sparseMatrixFeaturesTrans)
  }

  it should "encode a sparse protobuf record with labels in RecordIO" in {
    val protobufRecord = ProtobufConverter.rowToProtobuf(sparseRowWithLabels,
      featuresColumnName, Some(labelColumnName))
    val recordIOByteArray = ProtobufConverter.byteArrayToRecordIOEncodedByteArray(
      protobufRecord.toByteArray)
    validateRecordIOEncoding(recordIOByteArray)
  }

  it should "encode a dense protobuf record with labels in RecordIO" in {
    val protobufRecord = ProtobufConverter.rowToProtobuf(denseRowWithLabels,
      featuresColumnName, Some(labelColumnName))
    val recordIOByteArray = ProtobufConverter.byteArrayToRecordIOEncodedByteArray(
      protobufRecord.toByteArray)
    validateRecordIOEncoding(recordIOByteArray)
  }

  it should "encode a sparse protobuf record without labels in RecordIO" in {
    val protobufRecord = ProtobufConverter.rowToProtobuf(sparseRowWithoutLabels,
      featuresColumnName, Some(labelColumnName))
    val recordIOByteArray = ProtobufConverter.byteArrayToRecordIOEncodedByteArray(
      protobufRecord.toByteArray)
    validateRecordIOEncoding(recordIOByteArray)
  }

  it should "encode a dense protobuf record without labels in RecordIO" in {
    val protobufRecord = ProtobufConverter.rowToProtobuf(denseRowWithoutLabels,
      featuresColumnName, Some(labelColumnName))
    val recordIOByteArray = ProtobufConverter.byteArrayToRecordIOEncodedByteArray(
      protobufRecord.toByteArray)
    validateRecordIOEncoding(recordIOByteArray)
  }

  it should "refuse to convert a row without a features column to protobuf" in {
    val schemaWithoutFeatures = new StructType().add(StructField("notFeatures", VectorType, true))
    val row : Row = new GenericRowWithSchema(Array(sparseFeatures), schemaWithoutFeatures)
    intercept[IllegalArgumentException] {
      ProtobufConverter.rowToProtobuf(row, featuresColumnName, Some(labelColumnName))
    }
  }

  private def validateDense(record: Record, denseVector: DenseVector) : Unit = {
    val recordValuesList = recordToValuesList(record)
    val denseVectorArray = denseVector.toArray
    assert(recordValuesList.size == denseVectorArray.length)
    for ((recordValue, value) <- recordValuesList zip denseVectorArray) {
      assert(value == recordValue)
    }
  }

  private def validateDense(record: Record, denseMatrix: DenseMatrix) : Unit = {
    val recordValuesList = recordToValuesList(record)
    val recordShape = recordToMatrixShape(record)
    assert(recordShape(0) == denseMatrix.numRows)
    assert(recordShape(1) == denseMatrix.numCols)

    // We always store in CSR, thus we need to pass isTransposed=true in order
    // to re-generate the original matrix
    val recordMatrix = new DenseMatrix(denseMatrix.numRows, denseMatrix.numCols,
      recordValuesList.asScala.toArray.map(_.toDouble), true)
    assert(recordMatrix.equals(denseMatrix))
  }

  private def validateSparse(record: Record, sparseVector: SparseVector) : Unit = {
    val recordValuesList = recordToValuesList(record)
    val recordKeysList = recordToKeysList(record)
    for ((recordValue, value) <- recordValuesList zip sparseVector.values) {
      assert(value == recordValue)
    }

    for ((recordIndex, index) <- recordKeysList zip sparseVector.indices) {
      assert(index == recordIndex)
    }

    assert(recordToShape(record) == sparseVector.size)
  }

  private def validateSparse(record: Record, sparseMatrix: SparseMatrix) : Unit = {
    val recordValuesList = recordToValuesList(record)
    val recordKeysList = matrixRecordToKeysList(record)
    val recordShape = recordToMatrixShape(record)
    val numRows = recordShape(0).toInt
    val numCols = recordShape(1).toInt

    // Read tuples (row, col, value). The encoding is:
    // row[i] = floor(key[i] / cols)
    // col[i] = key[i] % cols
    val coordinates : java.util.List[scala.Tuple3[Int, Int, Double]] = new java.util.ArrayList()
    for(idx <- (0 to recordValuesList.size - 1)) {
      coordinates.add((recordKeysList(idx).toInt / numCols, recordKeysList(idx).toInt % numCols,
        recordValuesList(idx).toDouble))
    }

    // Reconstruct SparseMatrix from the coordinates
    val newSparseMatrix = SparseMatrix.fromCOO(numRows, numCols, coordinates.toIterable)

    assert(sparseMatrix.equals(newSparseMatrix))
  }

  private def recordToKeysList(record: Record) : java.util.List[java.lang.Long] = {
    validateSparseRecord(record)
    getFeaturesTensorFromRecord(record).getKeysList
  }

  private def matrixRecordToKeysList(record: Record) : java.util.List[java.lang.Long] = {
    getFeaturesTensorFromRecord(record).getKeysList
  }

  private def recordToShape(record: Record) : java.lang.Long = {
    validateSparseRecord(record)
    getFeaturesTensorFromRecord(record).getShape(0)
  }

  private def recordToMatrixShape(record: Record) : java.util.List[java.lang.Long] = {
    getFeaturesTensorFromRecord(record).getShapeList
  }

  private def recordToLabel(record: Record) : Float = {
    record.getLabel(0).getValue.getFloat32Tensor.getValues(0)
  }

  private def recordToValuesList(record: Record) : java.util.List[java.lang.Float] = {
    getFeaturesTensorFromRecord(record).getValuesList
  }

  private def validateSparseRecord(record: Record) : Unit = {
    val featuresTensor = getFeaturesTensorFromRecord(record)
    require(featuresTensor.getShapeCount == 1,
      "Sparse record shape should have exactly one entry for sparse record " +
        "length, not " + featuresTensor.getShapeCount)
    val keyCount = featuresTensor.getKeysCount
    val valueCount = featuresTensor.getValuesCount
    require(keyCount >= 1,
      "Sparse record key count is %s, but must be a non-negative integer".format(keyCount))
    require(keyCount == valueCount,
      ("Sparse record key count is %s, value count is" +
        " %s -- these should be equal").format(keyCount, valueCount))
  }

  private def getFeaturesTensorFromRecord(record: Record) : RecordProto2.Float32Tensor = {
    val featuresList = record.getFeaturesList
    for (featureEntry: RecordProto2.MapEntry <- featuresList) {
      if (featureEntry.getKey.equals(ValuesIdentifierString)) {
        return featureEntry.getValue.getFloat32Tensor
      }
    }
    throw new IllegalArgumentException("Record does not have a features tensor.")
  }

  private def validateRecordIOEncoding(byteArray: Array[Byte]) : Unit = {
    val buffer = ByteBuffer.wrap(byteArray).order(ByteOrder.LITTLE_ENDIAN)
    val magicNumber = buffer.getInt()
    ProtobufConverter.validateMagicNumber(magicNumber)

    // Second int should be equal in length to body of byte array.
    val recordLength = buffer.getInt
    val paddingNeeded = ProtobufConverter.paddingCount(recordLength)
    assert(8 + recordLength + paddingNeeded == byteArray.length)
  }
}
