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
import json
import py4j

from abc import ABCMeta

from pyspark import SparkContext
from pyspark.ml.common import _java2py
from pyspark.ml.wrapper import JavaWrapper
from pyspark.sql.types import StructType


class SageMakerJavaWrapper(JavaWrapper):

    __metaclass__ = ABCMeta
    _wrapped_class = None

    def __init__(self):
        super(SageMakerJavaWrapper, self).__init__()
        self.java_obj = None

    def _py2j(self, arg):
        if isinstance(arg, dict):
            return ScalaMap(arg)._to_java()
        elif isinstance(arg, list):
            return ScalaList(arg)._to_java()
        elif isinstance(arg, StructType):
            return JavaWrapper._new_java_obj(
                "org.apache.spark.sql.types.DataType.fromJson", json.dumps(arg.jsonValue())
            )
        elif isinstance(arg, SageMakerJavaWrapper):
            return arg._to_java()
        else:
            return arg

    def _new_java_obj(self, java_class, *args):
        """
        Creates a java object. We convert SageMakerJavaClass arguments
        to their java versions and then hand over to JavaWrapper

        :param java_class: Java ClassName
        :param args: constructor arguments
        :return: Java Instance
        """

        java_args = []
        for arg in args:
            java_args.append(self._py2j(arg))

        return JavaWrapper._new_java_obj(java_class, *java_args)

    def _call_java(self, name, *args):
        """
        Call a Java method in our Wrapped Class
        :param name: method name
        :param args: method arguments
        :return: java method return value converted to a python object
        """

        # call the base class method first to do the actual method calls
        # then we just need to call _from_java() to convert complex types to
        # python objects
        java_args = []
        for arg in args:
            java_args.append(self._py2j(arg))

        java_value = super(SageMakerJavaWrapper, self)._call_java(name, *java_args)
        return SageMakerJavaWrapper._from_java(java_value)

    def _to_java(self):
        if self._java_obj is None:
            self._java_obj = self._new_java_obj(self._wrapped_class)
        return self._java_obj

    @classmethod
    def _from_java(cls, java_object):

        # primitives and spark data types are converted automatically by
        # _java2py(), in those cases there is nothing to do
        if type(java_object) != py4j.java_gateway.JavaObject:
            return java_object

        # construct a mapping of our python wrapped classes to
        # java/scala classes
        wrapped_classes = {}
        for cls in SageMakerJavaWrapper.__subclasses__():
            wrapped_classes[cls._wrapped_class] = cls

        class_name = java_object.getClass().getName()

        # SageMakerJavaWrapper classes know how to convert themselves from a Java Object
        # otherwise hand over to _java2py and hope for the best.
        if class_name in wrapped_classes:
            return wrapped_classes[class_name]._from_java(java_object)
        elif class_name.startswith("scala.None"):
            return None
        else:
            sc = SparkContext._active_spark_context
            return _java2py(sc, java_object)


class Option(SageMakerJavaWrapper):

    _wrapped_class = "scala.Some"

    def __init__(self, value):
        self.value = value

    @classmethod
    def empty(cls):
        return JavaWrapper._new_java_obj("scala.Option.empty")

    def _to_java(self):
        if self.value is None:
            _java_obj = self._new_java_obj(
                "scala.Option.empty"
            )
            return _java_obj
        else:
            _java_obj = self._new_java_obj(
                "scala.Some",
                self.value)
            return _java_obj

    @classmethod
    def _from_java(cls, JavaObject):
        # Options are just wrappers around another object, strip the option layer
        # and convert the actual java object
        java_object = JavaObject.get()
        return SageMakerJavaWrapper._from_java(java_object)


class ScalaMap(SageMakerJavaWrapper):

    _wrapped_class = "scala.collection.immutable.HashMap"

    def __init__(self, dictionary):
        self.dictionary = dictionary

    def _to_java(self):
        map = self._new_java_obj(ScalaMap._wrapped_class)
        for k, v in self.dictionary.items():
            tuple = self._new_java_obj("scala.Tuple2", k, v)
            # equivalent to calling  map + (k, v)  in scala.
            # $plus() is the + operator.
            map = getattr(map, "$plus")(tuple)

        return map

    def _from_java(self):
        raise NotImplementedError()


# wrap scala.collection.immutable.List
class ScalaList(SageMakerJavaWrapper):

    _wrapped_class = "scala.collection.immutable.List"

    def __init__(self, p_list):
        self.p_list = p_list

    def _to_java(self):
        # Since py4j cannot deal with scala list directly
        # we convert to scala listmap as an intermediate step
        s_list = self._new_java_obj("scala.collection.immutable.ListMap")
        for key, elem in enumerate(self.p_list):
            tuple = self._new_java_obj("scala.Tuple2", key, elem)
            s_list = getattr(s_list, "$plus")(tuple)
        s_list = s_list.values().toList()

        return s_list

    def _from_java(self):
        raise NotImplementedError()
