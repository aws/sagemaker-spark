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

import java.io.ByteArrayOutputStream

import org.apache.hadoop.fs.Path
import org.apache.hadoop.io.{BytesWritable, NullWritable}
import org.apache.hadoop.mapreduce.{RecordWriter, TaskAttemptContext}

import org.apache.spark.sql.Row
import org.apache.spark.sql.catalyst.{CatalystTypeConverters, InternalRow}
import org.apache.spark.sql.execution.datasources.OutputWriter
import org.apache.spark.sql.types.StructType

/**
  * Writes rows of labeled vectors to Amazon protobuf Records encoded in RecordIO
  *
  * By default, writes a label column of Doubles named "label" and a features column
  * of Vector[Double]s named "features" to protobuf.
  *
  * These column names can be reassigned with the option "labelColumnName" and "featuresColumnName".
  *
  * To write records from a DataFrame in this file format, run
  * <code>dataframe.save
  *   .format("sagemaker")
  *   .option("labelColumnName", "myLabelColumn")
  *   .option("featuresColumnName", "myFeaturesColumn")
  *   .save("my_output_path")</code>
  *
  * @see [[https://aws.amazon.com/sagemaker/latest/dg/cdf-training.html/]] for more information on
  *      the Amazon Record data format.
  * @see [[https://mxnet.incubator.apache.org/architecture/note_data_loading.html]] for more
  *      information on recordIO
  */

class SageMakerProtobufWriter(path : String, context : TaskAttemptContext, dataSchema: StructType,
                              options: Map[String, String] = Map()) extends OutputWriter {

  private[protobuf] lazy val recordWriter: RecordWriter[NullWritable, BytesWritable] = {
    new RecordIOOutputFormat() {
      override def getDefaultWorkFile(context: TaskAttemptContext, extension: String): Path = {
        new Path(path)
      }
    }.getRecordWriter(context)
  }

  private val byteArrayOutputStream = new ByteArrayOutputStream()

  private val converter: InternalRow => Row =
    CatalystTypeConverters.createToScalaConverter(dataSchema).asInstanceOf[InternalRow => Row]

  /**
    * Writes a row to an underlying RecordWriter
    *
    * @param row the Row to be written as Amazon Records
    */
  def write(row: InternalRow): Unit = {
    write(converter(row))
  }

  /**
    * Writes a row to an underlying RecordWriter
    *
    * @param row the Row to be written as Amazon Records.
    */
  def write(row: Row): Unit = {
    val labelColumnName = options.getOrElse("labelColumnName", "label")
    val featuresColumnName = options.getOrElse("featuresColumnName", "features")

    val record = ProtobufConverter.rowToProtobuf(row, featuresColumnName, Some(labelColumnName))
    record.writeTo(byteArrayOutputStream)

    recordWriter.write(NullWritable.get(), new BytesWritable(byteArrayOutputStream.toByteArray))
    byteArrayOutputStream.reset()
  }

  override def close(): Unit = {
    recordWriter.close(context)
  }
}
