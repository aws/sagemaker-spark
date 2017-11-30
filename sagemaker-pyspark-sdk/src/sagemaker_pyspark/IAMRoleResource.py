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


class IAMRoleResource(SageMakerJavaWrapper):
    """
    References an IAM Role.
    """
    __metaclass__ = ABCMeta
    _wrapped_class = "com.amazonaws.services.sagemaker.sparksdk.IAMRoleResource"

    @classmethod
    def _from_java(cls, JavaObject):
        class_name = JavaObject.getClass().getName().split(".")[-1]

        if class_name == "IAMRole":
            return IAMRole._from_java(JavaObject)
        else:
            return None


class IAMRole(IAMRoleResource):
    """
    Specifies an IAM Role by  ARN or Name.

    Args:
        role (str): IAM Role Name or ARN.
    """
    _wrapped_class = "com.amazonaws.services.sagemaker.sparksdk.IAMRole"

    def __init__(self, role):
        self.role = role
        self._java_obj = None

    def _to_java(self):
        if self._java_obj is None:
            self._java_obj = self._new_java_obj(
                IAMRole._wrapped_class,
                self.role)

        return self._java_obj

    @classmethod
    def _from_java(cls, JavaObject):
        sc = SparkContext._active_spark_context
        role = _java2py(sc, JavaObject.role())
        return IAMRole(role)


class IAMRoleFromConfig(IAMRoleResource):
    """
    Gets an IAM role from Spark config

    Args:
        configKey (str): key in Spark config corresponding to IAM Role ARN.
    """
    _wrapped_class = "com.amazonaws.services.sagemaker.sparksdk.IAMRoleFromConfig"

    def __init__(self, configKey="com.amazonaws.services.sagemaker.sparksdk.sagemakerrole"):
        self.configKey = configKey
        self._java_obj = None

    def _to_java(self):
        if self._java_obj is None:
            self._java_obj = self._new_java_obj(
                IAMRole._wrapped_class,
                self.configKey)

        return self._java_obj

    @classmethod
    def _from_java(cls, JavaObject):
        sc = SparkContext._active_spark_context
        configKey = _java2py(sc, JavaObject.configKey())
        return IAMRoleFromConfig(configKey)
