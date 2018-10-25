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

from sagemaker_pyspark import SageMakerJavaWrapper


class NamePolicy(SageMakerJavaWrapper):
    """
    Provides names for SageMaker entities created during fit in
    :class:`~sagemaker_pyspark.JavaSageMakerEstimator`.
    """
    __metaclass__ = ABCMeta
    _wrapped_class = "com.amazonaws.services.sagemaker.sparksdk.NamePolicy"


class RandomNamePolicy(NamePolicy):
    """
    Provides random, unique SageMaker entity names that begin with the specified prefix.

    Args:
        prefix (str): The common name prefix for all SageMaker entities named with this NamePolicy.
    """

    _wrapped_class = "com.amazonaws.services.sagemaker.sparksdk.RandomNamePolicy"

    def __init__(self, prefix=""):
        self.prefix = prefix
        self._java_obj = None

    def _to_java(self):
        if self._java_obj is None:
            self._java_obj = self._new_java_obj(RandomNamePolicy._wrapped_class, self.prefix)
        return self._java_obj

    def _from_java(cls, JavaObject):
        pass


class CustomNamePolicy(NamePolicy):
    """
    Provides custom SageMaker entity names.

    Args:
        trainingJobName (str): The name of the SageMaker entity
        modelName (str): The model name of the SageMaker entity
        endpointConfigName (str): The endpoint config name of the SageMaker entity
        endpointName (str): The endpoint name of the SageMaker entity
    """

    _wrapped_class = "com.amazonaws.services.sagemaker.sparksdk.CustomNamePolicy"

    def __init__(self, trainingJobName, modelName, endpointConfigName, endpointName):
        self.trainingJobName = trainingJobName
        self.modelName = modelName
        self.endpointConfigName = endpointConfigName
        self.endpointName = endpointName
        self._java_obj = None

    def _to_java(self):
        if self._java_obj is None:
            self._java_obj = self._new_java_obj(CustomNamePolicy._wrapped_class,
                                                self.trainingJobName, self.modelName,
                                                self.endpointConfigName, self.endpointName)
        return self._java_obj

    def _from_java(cls, JavaObject):
        pass


class CustomNamePolicyWithTimeStampSuffix(NamePolicy):
    """
    Provides custom SageMaker entity names with timestamp suffix.

    Args:
        trainingJobName (str): The job name of the SageMaker entity
        modelName (str): The model name of the SageMaker entity
        endpointConfigName (str): The endpoint config name of the SageMaker entity
        endpointName (str): The endpoint name of the SageMaker entity
    """

    _wrapped_class = "com.amazonaws.services.sagemaker.sparksdk.CustomNamePolicyWithTimeStampSuffix"

    def __init__(self, trainingJobName, modelName, endpointConfigName, endpointName):
        self.trainingJobName = trainingJobName
        self.modelName = modelName
        self.endpointConfigName = endpointConfigName
        self.endpointName = endpointName
        self._java_obj = None

    def _to_java(self):
        if self._java_obj is None:
            self._java_obj = self._new_java_obj(CustomNamePolicyWithTimeStampSuffix._wrapped_class,
                                                self.trainingJobName, self.modelName,
                                                self.endpointConfigName, self.endpointName)
        return self._java_obj

    def _from_java(cls, JavaObject):
        pass


class NamePolicyFactory(SageMakerJavaWrapper):
    """
    Creates a NamePolicy upon a call to createNamePolicy
    :class:`~sagemaker_pyspark.JavaSageMakerEstimator`.
    """
    __metaclass__ = ABCMeta
    _wrapped_class = "com.amazonaws.services.sagemaker.sparksdk.NamePolicyFactory"

    def createNamePolicy(self):
        return self._call_java("createNamePolicy")


class RandomNamePolicyFactory(NamePolicyFactory):
    """
    Creates a RandomNamePolicy upon a call to createNamePolicy

    Args:
        prefix (str): The common name prefix for all SageMaker entities named with this NamePolicy.
    """

    _wrapped_class = "com.amazonaws.services.sagemaker.sparksdk.RandomNamePolicyFactory"

    def __init__(self, prefix=""):
        self.prefix = prefix
        self._java_obj = None

    def _to_java(self):
        if self._java_obj is None:
            self._java_obj = self._new_java_obj(RandomNamePolicyFactory._wrapped_class,
                                                self.prefix)
        return self._java_obj

    def _from_java(cls, JavaObject):
        pass

    def createNamePolicy(self):
        return self._call_java("createNamePolicy", self.prefix)


class CustomNamePolicyFactory(NamePolicyFactory):
    """
    Creates a CustomNamePolicy upon a call to createNamePolicy

    Args:
        trainingJobName (str): The job name of the SageMaker entity with this NamePolicy.
        modelName (str): The job name of the SageMaker entity with this NamePolicy.
        endpointConfigName (str): The job name of the SageMaker entity with this NamePolicy.
        endpointName (str): The job name of the SageMaker entity with this NamePolicy.
    """

    _wrapped_class = "com.amazonaws.services.sagemaker.sparksdk.CustomNamePolicyFactory"

    def __init__(self, trainingJobName, modelName, endpointConfigName, endpointName):
        self.trainingJobName = trainingJobName
        self.modelName = modelName
        self.endpointConfigName = endpointConfigName
        self.endpointName = endpointName
        self._java_obj = None

    def _to_java(self):
        if self._java_obj is None:
            self._java_obj = self._new_java_obj(CustomNamePolicyFactory._wrapped_class,
                                                self.trainingJobName, self.modelName,
                                                self.endpointConfigName, self.endpointName)
        return self._java_obj

    def _from_java(cls, JavaObject):
        pass

    def createNamePolicy(self):
        return self._call_java("createNamePolicy", self.trainingJobName,
                               self.modelName, self.endpointConfigName, self.endpointName)


class CustomNamePolicyWithTimeStampSuffixFactory(NamePolicyFactory):
    """
    Creates a CustomNamePolicyFactoryWithTimeStampSuffix upon a call to createNamePolicy

    Args:
        trainingJobName (str): The job name of the SageMaker entity with this NamePolicy.
        modelName (str): The job name of the SageMaker entity with this NamePolicy.
        endpointConfigName (str): The job name of the SageMaker entity with this NamePolicy.
        endpointName (str): The job name of the SageMaker entity with this NamePolicy.
    """

    _wrapped_class = \
        "com.amazonaws.services.sagemaker.sparksdk.CustomNamePolicyWithTimeStampSuffixFactory"

    def __init__(self, trainingJobName, modelName, endpointConfigName, endpointName):
        self.trainingJobName = trainingJobName
        self.modelName = modelName
        self.endpointConfigName = endpointConfigName
        self.endpointName = endpointName
        self._java_obj = None

    def _to_java(self):
        if self._java_obj is None:
            self._java_obj = self._new_java_obj(
                CustomNamePolicyWithTimeStampSuffixFactory._wrapped_class,
                self.trainingJobName, self.modelName,
                self.endpointConfigName, self.endpointName)
        return self._java_obj

    def _from_java(cls, JavaObject):
        pass

    def createNamePolicy(self):
        return self._call_java("createNamePolicy", self.trainingJobName,
                               self.modelName, self.endpointConfigName, self.endpointName)
