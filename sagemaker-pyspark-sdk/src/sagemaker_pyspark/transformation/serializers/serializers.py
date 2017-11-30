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
from sagemaker_pyspark import SageMakerJavaWrapper, Option


class RequestRowSerializer(SageMakerJavaWrapper):
    __metaclass__ = ABCMeta

    def setSchema(self, schema):
        """
        Sets the rowSchema for this RequestRowSerializer.

        Args:
            schema (StructType): the schema that this RequestRowSerializer will use.
        """
        self._call_java("setSchema", schema)


class UnlabeledCSVRequestRowSerializer(RequestRowSerializer):
    """
    Serializes according to the current implementation of the scoring service.

    Args:
        schema (StructType): tbd
        featuresColumnName (str): name of the features column.
    """
    _wrapped_class = "com.amazonaws.services.sagemaker.sparksdk.transformation." \
                     "serializers.UnlabeledCSVRequestRowSerializer"

    def __init__(self, schema=None, featuresColumnName="features"):
        self.schema = schema
        self.featuresColumnName = featuresColumnName

        self._java_obj = None

    def _to_java(self):
        if self._java_obj is None:
            self._java_obj = self._new_java_obj(
                UnlabeledCSVRequestRowSerializer._wrapped_class,
                Option(self.schema),
                self.featuresColumnName)

        return self._java_obj

    @classmethod
    def _from_java(cls, JavaObject):
        raise NotImplementedError()


class ProtobufRequestRowSerializer(RequestRowSerializer):
    """
    A RequestRowSerializer for converting labeled rows to SageMaker Protobuf-in-recordio request
    data.

    Args:
        schema (StructType): The schema of Rows being serialized. This parameter is optional as
            the schema may not be known when this serializer is constructed.
    """
    _wrapped_class = "com.amazonaws.services.sagemaker.sparksdk.transformation." \
                     "serializers.ProtobufRequestRowSerializer"

    def __init__(self, schema=None, featuresColumnName="features"):
        self.schema = schema
        self.featuresColumnName = featuresColumnName

        self._java_obj = None

    def _to_java(self):
        if self._java_obj is None:
            self._java_obj = self._new_java_obj(
                ProtobufRequestRowSerializer._wrapped_class,
                Option(self.schema),
                self.featuresColumnName)

        return self._java_obj

    @classmethod
    def _from_java(cls, JavaObject):
        raise NotImplementedError()


class LibSVMRequestRowSerializer(RequestRowSerializer):
    """
    Extracts a label column and features column from a Row and serializes as a LibSVM record.

    Each Row must contain a Double column and a Vector column containing the label and features
    respectively. Row field indexes for the label and features are obtained by looking up the
    index of labelColumnName and featuresColumnName respectively in the specified schema.

    A schema must be specified before this RequestRowSerializer can be used by a client. The
    schema is set either on instantiation of this RequestRowSerializer or by
    :meth:`RequestRowSerializer.setSchema`.

    Args:
        schema (StructType): The schema of Rows being serialized. This parameter is optional as
            the schema may not be known when this serializer is constructed.
        labelColumnName (str): The name of the label column.
        featuresColumnName (str): The name of the features column.

    """
    _wrapped_class = "com.amazonaws.services.sagemaker.sparksdk.transformation." \
                     "serializers.LibSVMRequestRowSerializer"

    def __init__(self, schema=None, labelColumnName="label", featuresColumnName="features"):
        self.schema = schema
        self.labelColumnName = labelColumnName
        self.featuresColumnName = featuresColumnName

        self._java_obj = None

    def _to_java(self):
        if self._java_obj is None:
            self._java_obj = self._new_java_obj(
                LibSVMRequestRowSerializer._wrapped_class,
                Option(self.schema),
                self.labelColumnName,
                self.featuresColumnName)

        return self._java_obj

    @classmethod
    def _from_java(cls, JavaObject):
        raise NotImplementedError()
