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

import java.util.concurrent.TimeUnit

/**
  * Injectable interface for interacting with System time.
  *
  * Used for testing purposes only.
  */
trait TimeProvider {
  def currentTimeMillis : Long
  def sleep(durationMillis : Long) : Unit
  def getElapsedTimeInSeconds(startTime: Long) : Long
}

/**
  * Implements TimeProvider using the Java System Time.
  */
class SystemTimeProvider extends TimeProvider {
  override def currentTimeMillis : Long = System.currentTimeMillis
  override def sleep(durationMillis : Long) : Unit = Thread.sleep(durationMillis)

  /**
    * Measures elapsed time, in seconds, for a given start time.
    *
    * @param startTime Starting time, in milliseconds, used to calculate elapsed time
    * @return Elapsed time in seconds
    */
  def getElapsedTimeInSeconds(startTime: Long): Long = {
    val elapsedTime = currentTimeMillis - startTime
    TimeUnit.MILLISECONDS.toSeconds(elapsedTime)
  }
}
