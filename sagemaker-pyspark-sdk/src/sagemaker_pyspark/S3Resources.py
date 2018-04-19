# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#   http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from abc import ABCMeta

from pyspark import SparkContext
from pyspark.ml.common import _java2py

from sagemaker_pyspark import SageMakerJavaWrapper


class S3Resource(SageMakerJavaWrapper):
    """
    An S3 Resource for SageMaker to use.
    """
    __metaclass__ = ABCMeta
    _wrapped_class = "com.amazonaws.services.sagemaker.sparksdk.S3Resource"

    @classmethod
    def _from_java(cls, JavaObject):
        class_name = JavaObject.getClass().getName().split(".")[-1]

        if class_name == "S3DataPath":
            return S3DataPath._from_java(JavaObject)
        else:
            return None


class S3AutoCreatePath(S3Resource, SageMakerJavaWrapper):
    """
    Defines an S3 location that will be auto-created at runtime.
    """
    _wrapped_class = "com.amazonaws.services.sagemaker.sparksdk.S3AutoCreatePath"

    @classmethod
    def _from_java(cls, JavaObject):
        return S3AutoCreatePath()


class S3DataPath(S3Resource, SageMakerJavaWrapper):
    """
    Represents a location within an S3 Bucket.

    Args:
        bucket (str): An S3 Bucket Name.
        objectPath (str): An S3 key or key prefix.
    """
    _wrapped_class = "com.amazonaws.services.sagemaker.sparksdk.S3DataPath"

    def __init__(self, bucket, objectPath):
        self.bucket = bucket
        self.objectPath = objectPath
        self._java_obj = self._new_java_obj(S3DataPath._wrapped_class, self.bucket, self.objectPath)

    def _to_java(self):
        return self._java_obj

    @classmethod
    def _from_java(cls, JavaObject):
        sc = SparkContext._active_spark_context

        bucket = _java2py(sc, JavaObject.bucket())
        object_path = _java2py(sc, JavaObject.objectPath())

        return S3DataPath(bucket, object_path)

    def toS3UriString(self):
        return self._call_java("toS3UriString")
