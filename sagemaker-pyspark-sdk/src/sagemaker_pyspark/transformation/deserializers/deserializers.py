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


class ResponseRowDeserializer(SageMakerJavaWrapper):

    __metaclass__ = ABCMeta


class ProtobufResponseRowDeserializer(ResponseRowDeserializer):
    """
    A :class:`~sagemaker_pyspark.transformation.deserializers.ResponseRowDeserializer`
    for converting SageMaker Protobuf-in-recordio response data to Spark rows.

    Args:
        schema (StructType): The schema of rows in the response.
    """
    _wrapped_class = "com.amazonaws.services.sagemaker.sparksdk.transformation." \
                     "deserializers.ProtobufResponseRowDeserializer"

    def __init__(self, schema, protobufKeys=None):
        self.schema = schema
        self.protobufKeys = protobufKeys
        self._java_obj = None

    def _to_java(self):
        if self._java_obj is None:
            self._java_obj = self._new_java_obj(
                ProtobufResponseRowDeserializer._wrapped_class,
                self.schema,
                self.protobufKeys)

        return self._java_obj

    @classmethod
    def _from_java(cls, JavaObject):
        raise NotImplementedError()


class KMeansProtobufResponseRowDeserializer(ResponseRowDeserializer):
    """
    Deserializes a Protobuf response from the Kmeans model image into Rows in a Spark DataFrame.

    Args:
        distance_to_cluster_column_name (str): name of the column of doubles indicating the
            distance to the nearest cluster from the input record.
        closest_cluster_column_name (str): name of the column of doubles indicating the label of
            the closest cluster for the input record.
    """
    _wrapped_class = "com.amazonaws.services.sagemaker.sparksdk.transformation." \
                     "deserializers.KMeansProtobufResponseRowDeserializer"

    def __init__(self, distance_to_cluster_column_name="distance_to_cluster",
                 closest_cluster_column_name="closest_cluster"):
        self.distance_to_cluster_column_name = distance_to_cluster_column_name
        self.closest_cluster_column_name = closest_cluster_column_name
        self._java_obj = None

    def _to_java(self):
        if self._java_obj is None:
            self._java_obj = self._new_java_obj(
                KMeansProtobufResponseRowDeserializer._wrapped_class,
                self.distance_to_cluster_column_name,
                self.closest_cluster_column_name)

        return self._java_obj

    @classmethod
    def _from_java(cls, JavaObject):
        raise NotImplementedError()


class PCAProtobufResponseRowDeserializer(ResponseRowDeserializer):
    """
    Deserializes a Protobuf response from the PCA model image to a Vector of Doubles containing
    the projection of the input vector.

    Args:
        projection_column_name (str): name of the column holding Vectors of Doubles representing
            the projected vectors.
    """
    _wrapped_class = "com.amazonaws.services.sagemaker.sparksdk.transformation." \
                     "deserializers.PCAProtobufResponseRowDeserializer"

    def __init__(self, projection_column_name="projection"):
        self.projection_column_name = projection_column_name
        self._java_obj = None

    def _to_java(self):
        if self._java_obj is None:
            self._java_obj = self._new_java_obj(
                PCAProtobufResponseRowDeserializer._wrapped_class,
                self.projection_column_name)

        return self._java_obj

    @classmethod
    def _from_java(cls, JavaObject):
        raise NotImplementedError()


class LDAProtobufResponseRowDeserializer(ResponseRowDeserializer):
    """
    Deserializes a Protobuf response from the LDA model image to a Vector of Doubles
    representing the topic mixture for the document represented by the input vector.

    Args:
        projection_column_name (str): name of the column holding Vectors of Doubles representing the
            topic mixtures for the documents.
    """
    _wrapped_class = "com.amazonaws.services.sagemaker.sparksdk.transformation." \
                     "deserializers.LDAProtobufResponseRowDeserializer"

    def __init__(self, projection_column_name="topic_mixture"):
        self.projection_column_name = projection_column_name
        self._java_obj = None

    def _to_java(self):
        if self._java_obj is None:
            self._java_obj = self._new_java_obj(
                LDAProtobufResponseRowDeserializer._wrapped_class,
                self.projection_column_name)

        return self._java_obj

    @classmethod
    def _from_java(cls, JavaObject):
        raise NotImplementedError()


class LinearLearnerBinaryClassifierProtobufResponseRowDeserializer(ResponseRowDeserializer):
    """
    Deserializes a Protobuf response from the LinearLearner model image with predictorType
    "binary_classifier" into Rows in a Spark Dataframe.

    Args:
        score_column_name (str): name of the column indicating the output score for
            the record.
        predicted_label_column_name(str): name of the column indicating the predicted
            label for the record.
    """
    _wrapped_class = "com.amazonaws.services.sagemaker.sparksdk.transformation." \
                     "deserializers.LinearLearnerBinaryClassifierProtobufResponseRowDeserializer"

    def __init__(self, score_column_name="score", predicted_label_column_name="predicted_label"):
        self.score_column_name = score_column_name
        self.predicted_label_column_name = predicted_label_column_name
        self._java_obj = None

    def _to_java(self):
        if self._java_obj is None:
            self._java_obj = self._new_java_obj(
                LinearLearnerBinaryClassifierProtobufResponseRowDeserializer._wrapped_class,
                self.score_column_name,
                self.predicted_label_column_name)

        return self._java_obj

    @classmethod
    def _from_java(cls, JavaObject):
        raise NotImplementedError()


class LinearLearnerMultiClassClassifierProtobufResponseRowDeserializer(ResponseRowDeserializer):
    """
    Deserializes a Protobuf response from the LinearLearner model image with predictorType
    "multiclass_classifier" into Rows in a Spark Dataframe.

    Args:
        score_column_name (str): name of the column indicating the output score for
            the record.
        predicted_label_column_name(str): name of the column indicating the predicted
            label for the record.
    """
    _wrapped_class = "com.amazonaws.services.sagemaker.sparksdk.transformation." \
                     "deserializers." \
                     "LinearLearnerMultiClassClassifierProtobufResponseRowDeserializer"

    def __init__(self, score_column_name="score", predicted_label_column_name="predicted_label"):
        self.score_column_name = score_column_name
        self.predicted_label_column_name = predicted_label_column_name
        self._java_obj = None

    def _to_java(self):
        if self._java_obj is None:
            self._java_obj = self._new_java_obj(
                LinearLearnerMultiClassClassifierProtobufResponseRowDeserializer._wrapped_class,
                self.score_column_name,
                self.predicted_label_column_name)

        return self._java_obj

    @classmethod
    def _from_java(cls, JavaObject):
        raise NotImplementedError()


class LinearLearnerRegressorProtobufResponseRowDeserializer(ResponseRowDeserializer):
    """
    Deserializes a Protobuf response from the LinearLearner model image with predictorType
    "regressor" into Rows in a Spark DataFrame.

    Args:
        score_column_name (str): name of the column of Doubles indicating the output score for
            the record.
    """
    _wrapped_class = "com.amazonaws.services.sagemaker.sparksdk.transformation." \
                     "deserializers.LinearLearnerRegressorProtobufResponseRowDeserializer"

    def __init__(self, score_column_name="score"):
        self.score_column_name = score_column_name
        self._java_obj = None

    def _to_java(self):
        if self._java_obj is None:
            self._java_obj = self._new_java_obj(
                LinearLearnerRegressorProtobufResponseRowDeserializer._wrapped_class,
                self.score_column_name)

        return self._java_obj

    @classmethod
    def _from_java(cls, JavaObject):
        raise NotImplementedError()


class LibSVMResponseRowDeserializer(ResponseRowDeserializer):
    """
    A :class:`~sagemaker_pyspark.transformation.deserializers.ResponseRowDeserializer`
    for converting LibSVM response data to labeled vectors.

    Args:
        dim (int): The vector dimension
        labelColumnName (str): The name of the label column
        featuresColumnName (str): The name of the features column
    """
    _wrapped_class = "com.amazonaws.services.sagemaker.sparksdk.transformation." \
                     "deserializers.LibSVMResponseRowDeserializer"

    def __init__(self, dim, labelColumnName, featuresColumnName):
        self.dim = dim
        self.labelColumnName = labelColumnName
        self.featuresColumnName = featuresColumnName
        self._java_obj = None

    def _to_java(self):
        if self._java_obj is None:
            self._java_obj = self._new_java_obj(
                self._wrapped_class, self.dim, self.labelColumnName, self.featuresColumnName)

        return self._java_obj


class XGBoostCSVRowDeserializer(ResponseRowDeserializer):
    """
    A :class:`~sagemaker_pyspark.transformation.deserializers.ResponseRowDeserializer`
    for converting a comma-delimited string of predictions to labeled Vectors.

    Args:
        prediction_column_name (str): the name of the output predictions column.
    """
    _wrapped_class = "com.amazonaws.services.sagemaker.sparksdk.transformation." \
                     "deserializers.XGBoostCSVRowDeserializer"

    def __init__(self, prediction_column_name="prediction"):
        self.prediction_column_name = prediction_column_name
        self._java_obj = None

    def _to_java(self):
        if self._java_obj is None:
            self._java_obj = self._new_java_obj(
                XGBoostCSVRowDeserializer._wrapped_class,
                self.prediction_column_name)

        return self._java_obj

    @classmethod
    def _from_java(cls, JavaObject):
        raise NotImplementedError()


class FactorizationMachinesBinaryClassifierDeserializer(ResponseRowDeserializer):
    """
    Deserializes a Protobuf response from the Factorization Machines model image with predictorType
    "binary_classifier" into Rows in a Spark Dataframe.

    Args:
        score_column_name (str): name of the column indicating the output score for
            the record.
        predicted_label_column_name(str): name of the column indicating the predicted
            label for the record.
    """
    _wrapped_class = "com.amazonaws.services.sagemaker.sparksdk.transformation." \
                     "deserializers.FactorizationMachinesBinaryClassifierDeserializer"

    def __init__(self, score_column_name="score", predicted_label_column_name="predicted_label"):
        self.score_column_name = score_column_name
        self.predicted_label_column_name = predicted_label_column_name
        self._java_obj = None

    def _to_java(self):
        if self._java_obj is None:
            self._java_obj = self._new_java_obj(
                FactorizationMachinesBinaryClassifierDeserializer._wrapped_class,
                self.score_column_name,
                self.predicted_label_column_name)

        return self._java_obj

    @classmethod
    def _from_java(cls, JavaObject):
        raise NotImplementedError()


class FactorizationMachinesRegressorDeserializer(ResponseRowDeserializer):
    """
    Deserializes a Protobuf response from the Factorization Machines model image with predictorType
    "regressor" into Rows in a Spark DataFrame.

    Args:
        score_column_name (str): name of the column of Doubles indicating the output score for
            the record.
    """
    _wrapped_class = "com.amazonaws.services.sagemaker.sparksdk.transformation." \
                     "deserializers.FactorizationMachinesRegressorDeserializer"

    def __init__(self, score_column_name="score"):
        self.score_column_name = score_column_name
        self._java_obj = None

    def _to_java(self):
        if self._java_obj is None:
            self._java_obj = self._new_java_obj(
                FactorizationMachinesRegressorDeserializer._wrapped_class,
                self.score_column_name)

        return self._java_obj

    @classmethod
    def _from_java(cls, JavaObject):
        raise NotImplementedError()
