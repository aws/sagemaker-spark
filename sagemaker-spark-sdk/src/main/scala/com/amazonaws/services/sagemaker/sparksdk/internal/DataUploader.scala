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
package com.amazonaws.services.sagemaker.sparksdk.internal

import java.io.{File, FileWriter, Writer}

import scala.collection.mutable.ListBuffer

import org.apache.hadoop.fs.{FileSystem, Path}

import org.apache.spark.sql.Dataset

import com.amazonaws.services.sagemaker.sparksdk.S3DataPath

/**
  * Uploads a DataFrame to S3 using the given data format and with the given options. Detects
  * if the application is being run on an EMR cluster to determine the S3 FileSystem scheme,
  * and whether to specify Training input data using a manifest file rather than an
  * S3 Prefix.
  *
  * [[com.amazonaws.services.sagemaker.sparksdk.SageMakerEstimator]]
  * depends on DataUploader to write DataFrames to S3 for training on
  * Amazon SageMaker.
  *
  * @param dataFormat the Spark-supported data format to write the DataFrame as, such as
  *                   "sagemaker" or "libsvm"
  * @param dataFormatOptions a map of options to configure the Spark DataFrameWriter corresponding
  *                          to the dataFormat.
  */
private[sparksdk] class DataUploader(final val dataFormat: String, final val dataFormatOptions:
collection.immutable.Map[String, String]) {

  /* Prefix for local file system temporary files containing a manifest of SageMaker training
  files */
  private final val ManifestTempFilePrefix = "sagemaker-manifest"

  /* Suffix for manifest s3 objects */
  private final val ManifestKeySuffix = ".manifest.txt"

  /* S3 metadata object key suffix, created by the hadoop filesystem */
  private final val HDFSFolderSuffix = "$folder$"

  /* Hadoop file system entry used to indicate success of a map-reduce file output committer */
  private final val HadoopSuccessFile = "_SUCCESS"

  /* Filesystem scheme to use when uploading to S3. */
  private final val S3FSScheme = "s3a"

  /* Filesystem scheme to use when uploading to S3 from EMR */
  private final val EMRFSScheme = "s3"

  /* Hadoop property that names the Filesystem implementation class. */
  private final val EmrfsProperty = "fs.s3.impl"

  /* When running on EMR, the property "fs.s3.impl" is this class name. */
  private final val EmrFSClassName = "EmrFileSystem"


  /**
    * Uploads a dataset to S3. If running on EMRFS, also uploads a manifest file containing a
    * list of the uploaded files.
    *
    * @param inputS3DataPath S3 URI to upload dataset to
    * @param dataset         Dataset to upload
    * @return a DataUploadResult containing an S3Path, either a ManifestDataUploadResult with an
    *         S3DataPath to a manifest file if a manifest was uploaded to S3 or an
    *         ObjectPrefixUploadResult with an S3DataPath to the S3 location
    *         where the dataset was uploaded if a manifest was not uploaded to S3.
    */
  def uploadData(inputS3DataPath: S3DataPath, dataset: Dataset[_]): DataUploadResult = {
    val inputDataBucket = inputS3DataPath.bucket
    val inputDataObjectPath = inputS3DataPath.objectPath

    val onEMR = usingEMRFS(dataset)
    val fsScheme = getFSScheme(onEMR)

    val inputURI = s"$fsScheme://$inputDataBucket/$inputDataObjectPath"
    writeData(dataset, inputURI)

    if (onEMR) {
      // Use manifest
      val inputPath = new Path(inputURI)
      val hadoopFS = getHadoopFilesystem(inputPath, dataset)
      val manifestKey = writeAndUploadManifest(inputPath, hadoopFS)
      ManifestDataUploadResult(S3DataPath(s"$inputDataBucket", s"$manifestKey"))
    } else {
      ObjectPrefixUploadResult(S3DataPath(inputDataBucket, inputDataObjectPath))
    }
  }

  private[internal] def getFSScheme(onEMR: Boolean) : String = {
    if (onEMR) EMRFSScheme else S3FSScheme
  }

  private[sparksdk] def writeData(dataset: Dataset[_], inputURI: String): Unit = {
    // Do not write _SUCCESS file.
    dataset.sparkSession.sparkContext.hadoopConfiguration.set(
      "mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")
    dataset.write.format(dataFormat).options(dataFormatOptions).save(inputURI)
  }

  private[internal] def usingEMRFS(dataset: Dataset[_]): Boolean = {
    val fileSystemClassName = dataset.sparkSession.sparkContext.
      hadoopConfiguration.get(EmrfsProperty)
    fileSystemClassName != null && fileSystemClassName.contains(EmrFSClassName)
  }

  private[internal] def getHadoopFilesystem(inputPath: Path, dataset: Dataset[_]): FileSystem = {
    inputPath.getFileSystem(dataset.sparkSession.sparkContext.hadoopConfiguration)
  }

  private[internal] def writeAndUploadManifest(sagemakerInput: Path, hadoopFS: FileSystem):
  String = {
    val tempFile = File.createTempFile(ManifestTempFilePrefix, ManifestTempFilePrefix)
    val manifestWriter = new FileWriter(tempFile)
    try {
      writeManifest(sagemakerInput, hadoopFS, manifestWriter)
    } finally {
      manifestWriter.close()
    }
    val manifestPath = new Path(s"${sagemakerInput.toUri.getPath}$ManifestKeySuffix")
    hadoopFS.copyFromLocalFile(true, new Path(tempFile.getAbsolutePath), manifestPath)
    manifestPath.toUri.getPath.stripPrefix("/") // Remove leading '/' to produce an s3 key
  }

  /**
    * Writes manifest entries for sagemaker training input files rooted in the input path.
    *
    * @param sagemakerInput The input path containing sagemaker data files
    * @param hadoopFS       The hadoop file system
    * @param manifestWriter The writer to write manifest entries to
    */
  private[internal] def writeManifest(sagemakerInput: Path,
                                      hadoopFS: FileSystem,
                                      manifestWriter: Writer): Unit = {
    val hadoopFSIterator = hadoopFS.listFiles(sagemakerInput, true)
    val manifestStringBuilder = new StringBuilder("[{\"prefix\": \"" + sagemakerInput + "/\"}, ")
    require(hadoopFSIterator.hasNext, s"No files found at $sagemakerInput.")
    val fileNames = ListBuffer.empty[String]
    while (hadoopFSIterator.hasNext) {
      val nextFile = hadoopFSIterator.next.getPath
      val fileName = nextFile.getName
      if (!isHadoopSpecialFile(fileName)) {
        fileNames.append("\"" + fileName + "\"")
      }
    }
    require(fileNames.nonEmpty, s"No non-hadoop files found at $sagemakerInput.")
    manifestStringBuilder.append(fileNames.mkString(", "))
    manifestWriter.write(manifestStringBuilder.append("]").toString)
  }

  private[internal] def isHadoopSpecialFile(fileName: String): Boolean = {
    fileName.endsWith(HDFSFolderSuffix) || fileName.equals(HadoopSuccessFile)
  }
}

sealed trait DataUploadResult {
  def s3DataPath: S3DataPath
}

case class ManifestDataUploadResult(override val s3DataPath: S3DataPath) extends DataUploadResult

case class ObjectPrefixUploadResult(override val s3DataPath: S3DataPath) extends DataUploadResult
