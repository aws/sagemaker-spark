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

package unit.com.amazonaws.services.sagemaker.sparksdk.transformation.serializers

import org.scalatest.{FlatSpec, Matchers}
import org.scalatest.mock.MockitoSugar

import org.apache.spark.ml.linalg.{DenseVector, SparseVector, SQLDataTypes}
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.types.{StructField, StructType}

import com.amazonaws.services.sagemaker.sparksdk.transformation.serializers.UnlabeledCSVRequestRowSerializer

class UnlabeledCSVRequestRowSerializerTests extends FlatSpec with Matchers with MockitoSugar {
  val schema: StructType =
    StructType(Array(StructField("features", SQLDataTypes.VectorType, nullable = false)))

  it should "serialize sparse vector" in {

    val vec = new SparseVector(100, Seq[Int](0, 10).toArray, Seq[Double](-100.0, 100.1).toArray)
    val row = new GenericRowWithSchema(values = Seq(vec).toArray, schema = schema)
    val rrs = new UnlabeledCSVRequestRowSerializer(Some(schema))
    val serialized = new String(rrs.serializeRow(row))
    val sparseString = "-100.0," + "0.0," * 9 + "100.1," + "0.0," * 88 + "0.0\n"
    assert (sparseString == serialized)
  }

  it should "serialize dense vector" in {
    val vec = new DenseVector(Seq(10.0, -100.0, 2.0).toArray)
    val row = new GenericRowWithSchema(values = Seq(vec).toArray, schema = schema)
    val rrs = new UnlabeledCSVRequestRowSerializer(Some(schema))
    val serialized = new String(rrs.serializeRow(row))
    assert("10.0,-100.0,2.0\n" == serialized)
  }
}
