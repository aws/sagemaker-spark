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

from sagemaker_pyspark import SageMakerJavaWrapper, Option


class SageMakerResourceCleanup(SageMakerJavaWrapper):

    _wrapped_class = "com.amazonaws.services.sagemaker.sparksdk.SageMakerResourceCleanup"

    def __init__(self, sagemakerClient, java_object=None):
        self.sagemakerClient = sagemakerClient
        if java_object is None:
            self._java_obj = self._new_java_obj(
                SageMakerResourceCleanup._wrapped_class, self.sagemakerClient)
        else:
            self._java_obj = java_object

    @classmethod
    def _from_java(cls, java_object):
        return SageMakerResourceCleanup(None, java_object)

    def deleteResources(self, created_resources):
        self._call_java("deleteResources", created_resources)

    def deleteEndpoint(self, name):
        self._call_java("deleteEndpoint", name)

    def deleteEndpointConfig(self, name):
        self._call_java("deleteEndpointConfig", name)

    def deleteModel(self, name):
        self._call_java("deleteModel", name)


class CreatedResources(SageMakerJavaWrapper):
    """Resources that may have been created during operation of the SageMaker Estimator and Model.

    Args:
        model_name (str): Name of the SageMaker Model that was created, or None if it wasn't
            created.
        endpoint_config_name (str): Name of the SageMaker EndpointConfig that was created, or None
            if it wasn't created.
        endpoint_name (str): Name of the SageMaker Endpoint that was created, or None if it wasn't
            created.
        java_object (:obj: `py4j.java_gateway.JavaObject`, optional): an existing CreatedResources
            java instance. If provided the other arguments are ignored.

    """

    _wrapped_class = "com.amazonaws.services.sagemaker.sparksdk.CreatedResources"

    def __init__(self, model_name=None,
                 endpoint_config_name=None,
                 endpoint_name=None,
                 java_object=None):
        if java_object is None:
            self.model_name = model_name
            self.endpoint_config_name = endpoint_config_name
            self.endpoint_name = endpoint_name
            self._java_obj = None
        else:
            self._java_obj = java_object
            self.model_name = self._call_java("modelName")
            self.endpoint_config_name = self._call_java("endpointConfigName")
            self.endpoint_name = self._call_java("endpointName")

    @classmethod
    def _from_java(cls, java_object):
        return CreatedResources(java_object=java_object)

    def _to_java(self):
        if self._java_obj is None:
            self._java_obj = self._new_java_obj(self._wrapped_class,
                                                Option(self.model_name),
                                                Option(self.endpoint_config_name),
                                                Option(self.endpoint_name))

        return self._java_obj
