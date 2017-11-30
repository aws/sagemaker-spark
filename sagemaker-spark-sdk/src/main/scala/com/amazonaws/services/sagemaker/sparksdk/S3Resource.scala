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

package com.amazonaws.services.sagemaker.sparksdk

import java.net.URI

/**
  * An S3 Resource for SageMaker to use.
  */
abstract class S3Resource

/**
  * Represents an S3 location defined by a Spark configuration key.
  * <p/>
  * The configuration key must either define a bucket name or an S3 URI of the form
  *{{{
  *   s3://bucket-name/prefix-path
  *}}}
  *
  * @param configKey The Spark configuration key to read the S3 location from
  */
case class S3PathFromConfig(val configKey : String =
                            "com.amazonaws.services.sagemaker.sparksdk.s3-data-bucket")
  extends S3Resource

/**
  * Defines an S3 location that will be auto-created at runtime.
  */
case class S3AutoCreatePath() extends S3Resource

/**
  * Represents a location within an S3 Bucket.
  *
  * @param bucket An S3 bucket
  * @param objectPath An S3 key or key prefix
  */
case class S3DataPath(val bucket : String, val objectPath : String) extends S3Resource {
  def toS3UriString : String = s"s3://$bucket/$objectPath"
}

object S3DataPath {

  /**
    * Constructs an S3DataPath from an S3 URI.
    *
    * @param uriString S3 URI in the form s3://bucket-name/prefix-path
    * @return S3DataPath object
    */
  def fromS3URI(uriString : String) : S3DataPath = {
    val uri = new URI(uriString)
    if (uri.getScheme.startsWith("s3")) {
      S3DataPath(uri.getAuthority, uri.getPath.stripPrefix("/"))
    } else {
      throw new IllegalArgumentException("Invalid scheme: " + uri.getScheme)
    }
  }
}
