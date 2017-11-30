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

import org.apache.hadoop.fs.FileStatus
import org.apache.hadoop.mapreduce.{Job, TaskAttemptContext}

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.execution.datasources.{FileFormat, OutputWriter, OutputWriterFactory}
import org.apache.spark.sql.sources.DataSourceRegister
import org.apache.spark.sql.types.StructType
/**
  * A Spark FileFormat for serializing Dataframes of labeled vectors to the Amazon Record
  * protobuf file format encoded in RecordIO.
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
class SageMakerProtobufFileFormat extends FileFormat with DataSourceRegister {

  override def inferSchema(sparkSession: SparkSession,
                           options: Map[String, String],
                           files: Seq[FileStatus]):
  Option[StructType] = {
    Option.empty
  }

  override def shortName(): String = "sagemaker"

  override def toString: String = "SageMaker"

  override def prepareWrite(
                             sparkSession: SparkSession,
                             job: Job,
                             options: Map[String, String],
                             dataSchema: StructType): OutputWriterFactory = {
    new OutputWriterFactory {
      override def newInstance(
                                path: String,
                                dataSchema: StructType,
                                context: TaskAttemptContext): OutputWriter = {
        new SageMakerProtobufWriter(path, context, dataSchema, options)
      }

      override def getFileExtension(context: TaskAttemptContext): String = {
        ".pbr"
      }
    }
  }
}
