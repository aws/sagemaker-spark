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

package object transformation {

  /**
    * SageMaker Transformation Content-Type HTTP Header Constants
    */
  object ContentTypes {

    /**
      * LibSVM format Content-Type Header
      *
      * @see http://www.csie.ntu.edu.tw/~cjlin/libsvm/faq.html
      */
    val TEXT_LIBSVM = "text/x-libsvm"

    /**
      * Hosting Service API Design: AWSALGO-556
      */
    val PROTOBUF = "application/x-recordio-protobuf"

    /**
      * JSON format Content-Type Header
      */
    val JSON = "application/json"

    /**
      * CSV format Content-Type Header
      */
    val CSV = "text/csv"
  }
}
