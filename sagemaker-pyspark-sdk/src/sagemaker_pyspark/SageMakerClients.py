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

from pyspark.ml.wrapper import JavaWrapper

from sagemaker_pyspark import SageMakerJavaWrapper


class SageMakerClients(object):

    class _S3DefaultClient(SageMakerJavaWrapper):
        _wrapped_class = "com.amazonaws.services.s3.AmazonS3ClientBuilder.defaultClient"

    class _STSDefaultClient(SageMakerJavaWrapper):
        _wrapped_class = "com.amazonaws.services.securitytoken." \
                        "AWSSecurityTokenServiceClientBuilder.defaultClient"

    class _SageMakerDefaultClient(SageMakerJavaWrapper):
        _wrapped_class = "com.amazonaws.services.sagemaker." \
                        "AmazonSageMakerClientBuilder.defaultClient"

    class _AmazonSageMaker(SageMakerJavaWrapper):
        _wrapped_class = "com.amazonaws.services.sagemaker.AmazonSageMakerClient"

        def __init__(self, javaObject):
            self._java_obj = javaObject

        @classmethod
        def _from_java(cls, javaObject):
            return SageMakerClients._AmazonSageMaker(javaObject)

    @classmethod
    def _getCreds(cls):
        return JavaWrapper._new_java_obj(
            "com.amazonaws.auth.DefaultAWSCredentialsProviderChain")

    @classmethod
    def create_s3_default_client(cls):
        return SageMakerClients._S3DefaultClient()

    @classmethod
    def create_sts_default_client(cls):
        return SageMakerClients._STSDefaultClient()

    @classmethod
    def create_sagemaker_client(cls):
        return SageMakerClients._SageMakerDefaultClient()
