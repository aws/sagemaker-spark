import numbers
from pyspark.ml.param import Params, Param, TypeConverters
from pyspark.ml.util import Identifiable

from sagemaker_pyspark import (SageMakerEstimatorBase, S3AutoCreatePath, Option, IAMRoleFromConfig,
                               EndpointCreationPolicy, SageMakerClients, RandomNamePolicyFactory)

from sagemaker_pyspark.transformation.serializers import ProtobufRequestRowSerializer
from sagemaker_pyspark.transformation.deserializers import (
    LinearLearnerBinaryClassifierProtobufResponseRowDeserializer,
    LinearLearnerRegressorProtobufResponseRowDeserializer)


class LinearLearnerParams(Params):

    feature_dim = Param(Params._dummy(), "feature_dim",
                        "The dimension of the input vectors. Must be > 0. ",
                        typeConverter=TypeConverters.toInt)

    mini_batch_size = Param(Params._dummy(), "mini_batch_size",
                            "The number of examples in a mini-batch. Must be > 0. ",
                            typeConverter=TypeConverters.toInt)

    epochs = Param(Params._dummy(), "epochs",
                   "The number of passes done over the training data. Must be > 0. ",
                   typeConverter=TypeConverters.toInt)

    predictor_type = Param(Params._dummy(), "predictor_type",
                           "Whether training is for binary classification or regression. "
                           "Supported options: 'binary_classifier', and 'regressor'. ",
                           typeConverter=TypeConverters.toString)

    use_bias = Param(Params._dummy(), "use_bias",
                     "Whether model should include bias. ",
                     typeConverter=TypeConverters.toString)

    num_models = Param(Params._dummy(), "num_models",
                       "Number of models to train in parallel. Must be > 0  or 'auto'. ",
                       typeConverter=TypeConverters.toString)

    num_calibration_samples = Param(Params._dummy(), "num_calibration_samples",
                                    "Number of samples to use from validation dataset for doing "
                                    "model calibration (finding the best threshold). "
                                    "Must be > 0.",
                                    typeConverter=TypeConverters.toInt)

    init_method = Param(Params._dummy(), "init_method",
                        "Initialization function for the model weights. "
                        "Supported options: 'uniform' and 'normal'. ",
                        typeConverter=TypeConverters.toString)

    init_scale = Param(Params._dummy(), "init_scale",
                       "Scale for init method uniform. Must be > 0. ",
                       typeConverter=TypeConverters.toFloat)

    init_sigma = Param(Params._dummy(), "init_sigma",
                       "Standard deviation for init method normal. Must be > 0. ",
                       typeConverter=TypeConverters.toFloat)

    init_bias = Param(Params._dummy(), "init_bias",
                      "Initial weight for bias. ",
                      typeConverter=TypeConverters.toFloat)

    optimizer = Param(Params._dummy(), "optimizer",
                      "Which optimizer is to be used. Supported options: "
                      "'sgd', 'adam'. ",
                      typeConverter=TypeConverters.toString)

    loss = Param(Params._dummy(), "loss",
                 "The loss function to apply. Supported options: "
                 "'logistic', 'squared_loss' and 'auto'.",
                 typeConverter=TypeConverters.toString)

    wd = Param(Params._dummy(), "wd",
               "The L2 regularization, i.e. the weight decay parameter. "
               "Must >= 0. ",
               typeConverter=TypeConverters.toFloat)

    l1 = Param(Params._dummy(), "l1",
               "The L1 regularization parameter. Use 0 for no L1 regularization. "
               "Must be in >= 0. ",
               typeConverter=TypeConverters.toFloat)

    momentum = Param(Params._dummy(), "momentum",
                     "Momentum parameter of sgd optimizer. Must be in range [0, 1). ",
                     typeConverter=TypeConverters.toFloat)

    learning_rate = Param(Params._dummy(), "learning_rate",
                          "The learning rate. The default 'auto' will depend upon the optimizer "
                          "selected. Must be > 0  or 'auto'. ",
                          typeConverter=TypeConverters.toString)

    beta_1 = Param(Params._dummy(), "beta_1",
                   "Parameter specific to adam optimizer. exponential decay rate for first moment "
                   "estimates. Must be in range [0, 1). ",
                   typeConverter=TypeConverters.toFloat)

    beta_2 = Param(Params._dummy(), "beta_2",
                   "Parameter specific to adam optimizer. exponential decay rate for second moment "
                   "estimates. Must be in range [0, 1). ",
                   typeConverter=TypeConverters.toFloat)

    bias_lr_mult = Param(Params._dummy(), "bias_lr_mult",
                         "Allows different learning rate for the bias term. "
                         "The actual learning rate for the bias is learning rate times "
                         "bias_lr_mult. Must be > 0. ",
                         typeConverter=TypeConverters.toFloat)

    bias_wd_mult = Param(Params._dummy(), "bias_wd_mult",
                         "Allows different learning rate for the bias term. "
                         "The actual L2 regularization weight for the bias is wd times "
                         "bias_wd_mult. Must be >= 0. ",
                         typeConverter=TypeConverters.toFloat)

    use_lr_scheduler = Param(Params._dummy(), "use_lr_scheduler",
                             "Whether to use a scheduler for the learning rate. ",
                             typeConverter=TypeConverters.toString)

    lr_scheduler_step = Param(Params._dummy(), "lr_scheduler_step",
                              "Parameter specific to lr_scheduler. "
                              "The number of steps between decreases of the learning rate. "
                              "Must be > 0. ",
                              typeConverter=TypeConverters.toInt)

    lr_scheduler_factor = Param(Params._dummy(), "lr_scheduler_factor",
                                "Parameter specific to lr_scheduler. "
                                "Every lr_scheduler_step the learning rate will decrease by this "
                                "quantity. Must be in (0, 1). ",
                                typeConverter=TypeConverters.toFloat)

    lr_scheduler_minimum_lr = Param(Params._dummy(), "lr_scheduler_minimum_lr",
                                    "Parameter specific to lr_scheduler. "
                                    "The learning rate will never decrease to a value lower than "
                                    "lr_scheduler_minimum_lr. Must be > 0. ",
                                    typeConverter=TypeConverters.toFloat)

    normalize_data = Param(Params._dummy(), "normalize_data",
                           "Whether to normalize the features before training to have "
                           "std_dev of 1.",
                           typeConverter=TypeConverters.toString)

    normalize_label = Param(Params._dummy(), "normalize_label",
                            "Whether regression label is normalized. "
                            "If set for classification, it will be ignored.",
                            typeConverter=TypeConverters.toString)

    unbias_data = Param(Params._dummy(), "unbias_data",
                        "Whether to unbias the features before training so that mean is 0. "
                        "By default data is unbiased if use_bias is set to true.",
                        typeConverter=TypeConverters.toString)

    unbias_label = Param(Params._dummy(), "unbias_label",
                         "Whether to unbias the labels before training so that mean is 0. "
                         "Only done for regrssion if use_bias is true. Otherwise will be ignored.",
                         typeConverter=TypeConverters.toString)

    num_point_for_scaler = Param(Params._dummy(), "num_point_for_scaler",
                                 "Number of data points to use for calcuating the "
                                 "normalizing / unbiasing terms. Must be > 0",
                                 typeConverter=TypeConverters.toInt)

    def getFeatureDim(self):
        return self.getOrDefault(self.feature_dim)

    def setFeatureDim(self, value):
        if value <= 0:
            raise ValueError("feature_dim must be > 0. Got %s" % value)
        self._set(feature_dim=value)

    def getMiniBatchSize(self):
        return self.getOrDefault(self.mini_batch_size)

    def setMiniBatchSize(self, size):
        if size <= 0:
            raise ValueError("mini_batch_size must be > 0. Got %s" % size)
        self._set(mini_batch_size=size)

    def getEpochs(self):
            return self.getOrDefault(self.epochs)

    def setEpochs(self, value):
        if value < 1:
            raise ValueError("Epochs must be > 0, got: %s" % value)
        self._set(epochs=value)

    def getUseBias(self):
        value = self.getOrDefault(self.use_bias)
        if value == 'True':
            return True
        else:
            return False

    def setUseBias(self, value):
        if value not in ('True', 'False'):
            raise ValueError("use_bias must be 'True' or 'False', got %s" % value)
        self._set(use_bias=value)

    def getNumModels(self):
        return self.getOrDefault(self.num_models)

    def setNumModels(self, value):
        if isinstance(value, numbers.Real) and value < 1:
            raise ValueError("num_models must be 'auto' or > 0, got: %s" % value)
        if value != 'auto' and int(value) < 1:
            raise ValueError("num_models must be 'auto' or > 0, got: %s" % value)
        self._set(num_models=str(value))

    def getNumCalibrationSamples(self):
        return self.getOrDefault(self.num_calibration_samples)

    def setNumCalibrationSamples(self, value):
        if value < 1:
            raise ValueError("num_calibration_samples must be > 0, got: %s" % value)
        self._set(num_calibration_samples=value)

    def getInitMethod(self):
        return self.getOrDefault(self.init_method)

    def setInitMethod(self, value):
        if value not in ('uniform', 'normal'):
            raise ValueError("init_method must be 'uniform' or 'normal', got: %s" % value)
        self._set(init_method=value)

    def getInitScale(self):
        return self.getOrDefault(self.init_scale)

    def setInitScale(self, value):
        if value <= 0:
            raise ValueError("init_scale must be > 0, got: %s" % value)
        self._set(init_scale=value)

    def getInitSigma(self):
        return self.getOrDefault(self.init_sigma)

    def setInitSigma(self, value):
        if value <= 0:
            raise ValueError("init_sigma must be > 0, got: %s" % value)
        self._set(init_sigma=value)

    def getInitBias(self):
        return self.getOrDefault(self.init_bias)

    def setInitBias(self, value):
        self._set(init_bias=value)

    def getOptimizer(self):
        return self.getOrDefault(self.optimizer)

    def setOptimizer(self, value):
        if value not in ('sgd', 'adam'):
            raise ValueError("optimizer must be 'sgd' or 'adam', got: %s" % value)
        self._set(optimizer=value)

    def getLoss(self):
        return self.getOrDefault(self.loss)

    def setLoss(self, value):
        if value not in ('logistic', 'squared_loss', 'auto'):
            raise ValueError("loss must be 'auto', 'logistic' or 'squared_loss', "
                             "got: %s" % value)
        self._set(loss=value)

    def getWd(self):
        return self.getOrDefault(self.wd)

    def setWd(self, value):
        if value < 0:
            raise ValueError("wd must be >= 0, got: %s" % value)
        self._set(wd=value)

    def getL1(self):
        return self.getOrDefault(self.l1)

    def setL1(self, value):
        if value < 0:
            raise ValueError("l1 must be >= 0, got: %s" % value)
        self._set(l1=value)

    def getMomentum(self):
        return self.getOrDefault(self.momentum)

    def setMomentum(self, value):
        if value >= 1 or value < 0:
            raise ValueError("momentum must be within [0, 1), got: %s" % value)
        self._set(momentum=value)

    def getLearningRate(self):
        return self.getOrDefault(self.learning_rate)

    def setLearningRate(self, value):
        if isinstance(value, numbers.Real) and value <= 0:
            raise ValueError("learning_rate must be 'auto' or > 0, got: %s" % value)
        if value != 'auto' and float(value) <= 0:
            raise ValueError("learning_rate must be 'auto' or > 0, got: %s" % value)
        self._set(learning_rate=str(value))

    def getBeta1(self):
        return self.getOrDefault(self.beta_1)

    def setBeta1(self, value):
        if value >= 1 or value < 0:
            raise ValueError("beta_1 must be within [0, 1), got: %s" % value)
        self._set(beta_1=value)

    def getBeta2(self):
        return self.getOrDefault(self.beta_2)

    def setBeta2(self, value):
        if value >= 1 or value < 0:
            raise ValueError("beta_2 must be within [0, 1), got: %s" % value)
        self._set(beta_2=value)

    def getBiasLrMult(self):
        return self.getOrDefault(self.bias_lr_mult)

    def setBiasLrMult(self, value):
        if value <= 0:
            raise ValueError("bias_lr_mult must be > 0, got: %s" % value)
        self._set(bias_lr_mult=value)

    def getBiasWdMult(self):
        return self.getOrDefault(self.bias_wd_mult)

    def setBiasWdMult(self, value):
        if value < 0:
            raise ValueError("bias_wd_mult must be >= 0, got: %s" % value)
        self._set(bias_wd_mult=value)

    def getUseLrScheduler(self):
        value = self.getOrDefault(self.use_lr_scheduler)
        if value == 'True':
            return True
        else:
            return False

    def setUseLrScheduler(self, value):
        if value not in ('True', 'False'):
            raise ValueError("use_lr_scheduler must be 'True' or 'False', got %s" % value)
        self._set(use_lr_scheduler=value)

    def getLrSchedulerStep(self):
        return self.getOrDefault(self.lr_scheduler_step)

    def setLrSchedulerStep(self, value):
        if value <= 0:
            raise ValueError("lr_scheduler_step must be > 0, got: %s" % value)
        self._set(lr_scheduler_step=value)

    def getLrSchedulerFactor(self):
        return self.getOrDefault(self.lr_scheduler_factor)

    def setLrSchedulerFactor(self, value):
        if value >= 1 or value <= 0:
            raise ValueError("lr_scheduler_factor must be in (0, 1), got: %s" % value)
        self._set(lr_scheduler_factor=value)

    def getLrSchedulerMinimumLr(self):
        return self.getOrDefault(self.lr_scheduler_minimum_lr)

    def setLrSchedulerMinimumLr(self, value):
        if value <= 0:
            raise ValueError("lr_scheduler_minimum_lr must be > 0, got: %s" % value)
        self._set(lr_scheduler_minimum_lr=value)

    def getNormalizeData(self):
        value = self.getOrDefault(self.normalize_data)
        if value == 'True':
            return True
        else:
            return False

    def setNormalizeData(self, value):
        if value not in ('True', 'False'):
            raise ValueError("normalize_data must be 'True' or 'False', got %s" % value)
        self._set(normalize_data=value)

    def getNormalizeLabel(self):
        value = self.getOrDefault(self.normalize_label)
        if value == 'True':
            return True
        else:
            return False

    def setNormalizeLabel(self, value):
        if value not in ('True', 'False'):
            raise ValueError("normalize_label must be 'True' or 'False', got %s" % value)
        self._set(normalize_label=value)

    def getUnbiasData(self):
        value = self.getOrDefault(self.unbias_data)
        if value == 'True':
            return True
        else:
            return False

    def setUnbiasData(self, value):
        if value not in ('True', 'False'):
            raise ValueError("unbias_data must be 'True' or 'False', got %s" % value)
        self._set(unbias_data=value)

    def getUnbiasLabel(self):
        value = self.getOrDefault(self.unbias_label)
        if value == 'True':
            return True
        else:
            return False

    def setUnbiasLabel(self, value):
        if value not in ('True', 'False'):
            raise ValueError("unbias_label must be 'True' or 'False', got %s" % value)
        self._set(unbias_label=value)

    def getNumPointForScaler(self):
        return self.getOrDefault(self.num_point_for_scaler)

    def setNumPointForScaler(self, value):
        if value <= 0:
            raise ValueError("numPointForScaler must be > 0, got: %s" % value)
        self._set(num_point_for_scaler=value)


class LinearLearnerBinaryClassifier(SageMakerEstimatorBase, LinearLearnerParams):
    """
    A :class:`~sagemaker_pyspark.SageMakerEstimator` that runs a Linear Learner training job in
    "binary classifier" mode in SageMaker and returns a :class:`~sagemaker_pyspark.SageMakerModel`
    that can be used to transform a DataFrame using the hosted Linear Learner model. The Linear
    Learner Binary Classifier is useful for classifying examples into one of two classes.

    Amazon SageMaker Linear Learner trains on RecordIO-encoded Amazon Record protobuf data.
    SageMaker pyspark writes a DataFrame to S3 by selecting a column of Vectors named "features"
    and, if present, a column of Doubles named "label". These names are configurable by passing a
    dictionary with entries in trainingSparkDataFormatOptions with key "labelColumnName" or
    "featuresColumnName", with values corresponding to the desired label and features columns.

    Inferences made against an Endpoint hosting a Linear Learner Binary classifier model contain
    a "score" field and a "predicted_label" field, both appended to the input DataFrame as Doubles.

    Args:
        sageMakerRole (IAMRole): The SageMaker TrainingJob and Hosting IAM Role. Used by
            SageMaker to access S3 and ECR Resources. SageMaker hosted Endpoint instances
            launched by this Estimator run with this role.
        trainingInstanceType (str): The SageMaker TrainingJob Instance Type to use.
        trainingInstanceCount (int): The number of instances of instanceType to run an
            SageMaker Training Job with.
        endpointInstanceType (str): The SageMaker Endpoint Config instance type.
        endpointInitialInstanceCount (int): The SageMaker Endpoint Config minimum number of
            instances that can be used to host modelImage.
        requestRowSerializer (RequestRowSerializer): Serializes Spark DataFrame Rows for
            transformation by Models built from this Estimator.
        responseRowDeserializer (ResponseRowDeserializer): Deserializes an Endpoint response into a
            series of Rows.
        trainingInputS3DataPath (S3Resource): An S3 location to upload SageMaker Training Job input
            data to.
        trainingOutputS3DataPath (S3Resource): An S3 location for SageMaker to store Training Job
            output data to.
        trainingInstanceVolumeSizeInGB (int): The EBS volume size in gigabytes of each instance.
        trainingProjectedColumns (List): The columns to project from the Dataset being fit before
            training. If an Optional.empty is passed then no specific projection will occur and
            all columns will be serialized.
        trainingChannelName (str): The SageMaker Channel name to input serialized Dataset fit
            input to.
        trainingContentType (str): The MIME type of the training data.
        trainingS3DataDistribution (str): The SageMaker Training Job S3 data distribution scheme.
        trainingSparkDataFormat (str): The Spark Data Format name used to serialize the Dataset
            being fit for input to SageMaker.
        trainingSparkDataFormatOptions (dict): The Spark Data Format Options used during
            serialization of the Dataset being fit.
        trainingInputMode (str): The SageMaker Training Job Channel input mode.
        trainingCompressionCodec (str): The type of compression to use when serializing the
            Dataset being fit for input to SageMaker.
        trainingMaxRuntimeInSeconds (int): A SageMaker Training Job Termination Condition
            MaxRuntimeInHours.
        trainingKmsKeyId (str): A KMS key ID for the Output Data Source.
        modelEnvironmentVariables (dict): The environment variables that SageMaker will set on the
            model container during execution.
        endpointCreationPolicy (EndpointCreationPolicy): Defines how a SageMaker Endpoint
            referenced by a SageMakerModel is created.
        sagemakerClient (AmazonSageMaker) Amazon SageMaker client. Used to send CreateTrainingJob,
            CreateModel, and CreateEndpoint requests.
        region (str): The region in which to run the algorithm. If not specified, gets the region
            from the DefaultAwsRegionProviderChain.
        s3Client (AmazonS3): Used to create a bucket for staging SageMaker Training Job
            input and/or output if either are set to S3AutoCreatePath.
        stsClient (AmazonSTS): Used to resolve the account number when creating staging
            input / output buckets.
        modelPrependInputRowsToTransformationRows (bool): Whether the transformation result on
            Models built by this Estimator should also include the input Rows. If true,
            each output Row is formed by a concatenation of the input Row with the corresponding
            Row produced by SageMaker Endpoint invocation, produced by responseRowDeserializer.
            If false, each output Row is just taken from responseRowDeserializer.
        deleteStagingDataAfterTraining (bool): Whether to remove the training data on s3 after
            training is complete or failed.
        namePolicyFactory (NamePolicyFactory): The NamePolicyFactory to use when naming SageMaker
            entities created during fit.
        uid (str): The unique identifier of this Estimator. Used to represent this stage in Spark
            ML pipelines.
       """
    _wrapped_class = \
        "com.amazonaws.services.sagemaker.sparksdk.algorithms.LinearLearnerBinaryClassifier"

    binary_classifier_model_selection_criteria = Param(
        Params._dummy(),
        "binary_classifier_model_selection_criteria",
        "Pick the model with best criteria from the validation dataset for predictor_type = "
        "binary_classifier.  Supported options: 'accuracy', 'f1', 'precision_at_target_recall',"
        " 'recall_at_target_precision' and 'cross_entropy_loss'. ",
        typeConverter=TypeConverters.toString)

    target_recall = Param(Params._dummy(), "target_recall",
                          "Applicable if binary_classifier_model_selection_criteria is "
                          "precision_at_target_recall. Must be in range [0, 1]. ",
                          typeConverter=TypeConverters.toFloat)

    target_precision = Param(Params._dummy(), "target_precision",
                             "Applicable if binary_classifier_model_selection_criteria is "
                             "recall_at_target_precision. Must be in range [0, 1]. ",
                             typeConverter=TypeConverters.toFloat)

    def __init__(
            self,
            trainingInstanceType,
            trainingInstanceCount,
            endpointInstanceType,
            endpointInitialInstanceCount,
            sagemakerRole=IAMRoleFromConfig(),
            requestRowSerializer=ProtobufRequestRowSerializer(),
            responseRowDeserializer=LinearLearnerBinaryClassifierProtobufResponseRowDeserializer(),
            trainingInputS3DataPath=S3AutoCreatePath(),
            trainingOutputS3DataPath=S3AutoCreatePath(),
            trainingInstanceVolumeSizeInGB=1024,
            trainingProjectedColumns=None,
            trainingChannelName="train",
            trainingContentType=None,
            trainingS3DataDistribution="ShardedByS3Key",
            trainingSparkDataFormat="sagemaker",
            trainingSparkDataFormatOptions=None,
            trainingInputMode="File",
            trainingCompressionCodec=None,
            trainingMaxRuntimeInSeconds=24*60*60,
            trainingKmsKeyId=None,
            modelEnvironmentVariables=None,
            endpointCreationPolicy=EndpointCreationPolicy.CREATE_ON_CONSTRUCT,
            sagemakerClient=SageMakerClients.create_sagemaker_client(),
            region=None,
            s3Client=SageMakerClients.create_s3_default_client(),
            stsClient=SageMakerClients.create_sts_default_client(),
            modelPrependInputRowsToTransformationRows=True,
            deleteStagingDataAfterTraining=True,
            namePolicyFactory=RandomNamePolicyFactory(),
            uid=None,
            javaObject=None):

        if trainingSparkDataFormatOptions is None:
            trainingSparkDataFormatOptions = {}

        if modelEnvironmentVariables is None:
            modelEnvironmentVariables = {}

        if uid is None:
            uid = Identifiable._randomUID()

        kwargs = locals().copy()
        del kwargs['self']

        super(LinearLearnerBinaryClassifier, self).__init__(**kwargs)

        default_params = {
            'predictor_type': 'binary_classifier'
        }

        self._setDefault(**default_params)

    def _get_java_obj(self, **kwargs):
        if 'javaObject' in kwargs and kwargs['javaObject'] is not None:
            return kwargs['javaObject']
        else:
            return self._new_java_obj(
                LinearLearnerBinaryClassifier._wrapped_class,
                kwargs['sagemakerRole'],
                kwargs['trainingInstanceType'],
                kwargs['trainingInstanceCount'],
                kwargs['endpointInstanceType'],
                kwargs['endpointInitialInstanceCount'],
                kwargs['requestRowSerializer'],
                kwargs['responseRowDeserializer'],
                kwargs['trainingInputS3DataPath'],
                kwargs['trainingOutputS3DataPath'],
                kwargs['trainingInstanceVolumeSizeInGB'],
                Option(kwargs['trainingProjectedColumns']),
                kwargs['trainingChannelName'],
                Option(kwargs['trainingContentType']),
                kwargs['trainingS3DataDistribution'],
                kwargs['trainingSparkDataFormat'],
                kwargs['trainingSparkDataFormatOptions'],
                kwargs['trainingInputMode'],
                Option(kwargs['trainingCompressionCodec']),
                kwargs['trainingMaxRuntimeInSeconds'],
                Option(kwargs['trainingKmsKeyId']),
                kwargs['modelEnvironmentVariables'],
                kwargs['endpointCreationPolicy'],
                kwargs['sagemakerClient'],
                Option(kwargs['region']),
                kwargs['s3Client'],
                kwargs['stsClient'],
                kwargs['modelPrependInputRowsToTransformationRows'],
                kwargs['deleteStagingDataAfterTraining'],
                kwargs['namePolicyFactory'],
                kwargs['uid']
            )

    def getBinaryClassifierModelSelectionCriteria(self):
        return self.getOrDefault(self.binary_classifier_model_selection_criteria)

    def setBinaryClassifierModelSelectionCriteria(self, value):
        if value not in ('accuracy', 'f1', 'precision_at_target_recall',
                         'recall_at_target_precision', 'cross_entropy_loss'):
            raise ValueError("binary_classifier_model_selection_criteria must be 'accuracy', 'f1', "
                             "'precision_at_target_recall','recall_at_target_precision',"
                             " 'cross_entropy_loss', got: %s" % value)
        self._set(binary_classifier_model_selection_criteria=value)

    def getTargetRecall(self):
        return self.getOrDefault(self.target_recall)

    def setTargetRecall(self, value):
        if value > 1 or value < 0:
            raise ValueError("target_recall must be within [0, 1], got: %s" % value)
        self._set(target_recall=value)

    def getTargetPrecision(self):
        return self.getOrDefault(self.target_precision)

    def setTargetPrecision(self, value):
        if value > 1 or value < 0:
            raise ValueError("target_precision must be within [0, 1], got: %s" % value)
        self._set(target_precision=value)

    @classmethod
    def _from_java(cls, javaObject):
        return LinearLearnerBinaryClassifier(sagemakerRole=None, javaObject=javaObject)


class LinearLearnerRegressor(SageMakerEstimatorBase, LinearLearnerParams):
    """
    A :class:`~sagemaker_pyspark.SageMakerEstimator` that runs a Linear Learner training job in
    "regressor" mode in SageMaker and returns a  :class:`~sagemaker_pyspark.SageMakerModel` that
    can be used to transform a DataFrame using the hosted Linear Learner model. The Linear Learner
    Regressor is useful for predicting a real-valued label from training examples.

    Amazon SageMaker Linear Learner trains on RecordIO-encoded Amazon Record protobuf data.
    SageMaker pyspark writes a DataFrame to S3 by selecting a column of Vectors named "features"
    and, if present, a column of Doubles named "label". These names are configurable by passing a
    dictionary with entries in trainingSparkDataFormatOptions with key "labelColumnName" or
    "featuresColumnName", with values corresponding to the desired label and features columns.

    For inference against a hosted Endpoint, the SageMakerModel returned by :meth :`fit()` by
    Linear Learner uses :class:`~sagemaker_pyspark.transformation
    .serializers.ProtobufRequestRowSerializer` to serialize Rows into RecordIO-encoded Amazon
    Record protobuf messages, by default selecting the column named "features" expected to contain
    a Vector of Doubles.

    Inferences made against an Endpoint hosting a Linear Learner Regressor model contain a "score"
    field appended to the input DataFrame as a Double.

    Args:
        sageMakerRole (IAMRole): The SageMaker TrainingJob and Hosting IAM Role. Used by
            SageMaker to access S3 and ECR Resources. SageMaker hosted Endpoint instances
            launched by this Estimator run with this role.
        trainingInstanceType (str): The SageMaker TrainingJob Instance Type to use.
        trainingInstanceCount (int): The number of instances of instanceType to run an
            SageMaker Training Job with.
        endpointInstanceType (str): The SageMaker Endpoint Config instance type.
        endpointInitialInstanceCount (int): The SageMaker Endpoint Config minimum number of
            instances that can be used to host modelImage.
        requestRowSerializer (RequestRowSerializer): Serializes Spark DataFrame Rows for
            transformation by Models built from this Estimator.
        responseRowDeserializer (ResponseRowDeserializer): Deserializes an Endpoint response into a
            series of Rows.
        trainingInputS3DataPath (S3Resource): An S3 location to upload SageMaker Training Job input
            data to.
        trainingOutputS3DataPath (S3Resource): An S3 location for SageMaker to store Training Job
            output data to.
        trainingInstanceVolumeSizeInGB (int): The EBS volume size in gigabytes of each instance.
        trainingProjectedColumns (List): The columns to project from the Dataset being fit before
            training. If an Optional.empty is passed then no specific projection will occur and
            all columns will be serialized.
        trainingChannelName (str): The SageMaker Channel name to input serialized Dataset fit
            input to.
        trainingContentType (str): The MIME type of the training data.
        trainingS3DataDistribution (str): The SageMaker Training Job S3 data distribution scheme.
        trainingSparkDataFormat (str): The Spark Data Format name used to serialize the Dataset
            being fit for input to SageMaker.
        trainingSparkDataFormatOptions (dict): The Spark Data Format Options used during
            serialization of the Dataset being fit.
        trainingInputMode (str): The SageMaker Training Job Channel input mode.
        trainingCompressionCodec (str): The type of compression to use when serializing the
            Dataset being fit for input to SageMaker.
        trainingMaxRuntimeInSeconds (int): A SageMaker Training Job Termination Condition
            MaxRuntimeInHours.
        trainingKmsKeyId (str): A KMS key ID for the Output Data Source.
        modelEnvironmentVariables (dict): The environment variables that SageMaker will set on the
            model container during execution.
        endpointCreationPolicy (EndpointCreationPolicy): Defines how a SageMaker Endpoint
            referenced by a SageMakerModel is created.
        sagemakerClient (AmazonSageMaker) Amazon SageMaker client. Used to send CreateTrainingJob,
            CreateModel, and CreateEndpoint requests.
        region (str): The region in which to run the algorithm. If not specified, gets the region
            from the DefaultAwsRegionProviderChain.
        s3Client (AmazonS3): Used to create a bucket for staging SageMaker Training Job
            input and/or output if either are set to S3AutoCreatePath.
        stsClient (AmazonSTS): Used to resolve the account number when creating staging
            input / output buckets.
        modelPrependInputRowsToTransformationRows (bool): Whether the transformation result on
            Models built by this Estimator should also include the input Rows. If true,
            each output Row is formed by a concatenation of the input Row with the corresponding
            Row produced by SageMaker Endpoint invocation, produced by responseRowDeserializer.
            If false, each output Row is just taken from responseRowDeserializer.
        deleteStagingDataAfterTraining (bool): Whether to remove the training data on s3 after
            training is complete or failed.
        namePolicyFactory (NamePolicyFactory): The NamePolicyFactory to use when naming SageMaker
            entities created during fit.
        uid (str): The unique identifier of this Estimator. Used to represent this stage in Spark
            ML pipelines.
           """
    _wrapped_class = "com.amazonaws.services.sagemaker.sparksdk.algorithms.LinearLearnerRegressor"

    def __init__(self,
                 trainingInstanceType,
                 trainingInstanceCount,
                 endpointInstanceType,
                 endpointInitialInstanceCount,
                 sagemakerRole=IAMRoleFromConfig(),
                 requestRowSerializer=ProtobufRequestRowSerializer(),
                 responseRowDeserializer=LinearLearnerRegressorProtobufResponseRowDeserializer(),
                 trainingInputS3DataPath=S3AutoCreatePath(),
                 trainingOutputS3DataPath=S3AutoCreatePath(),
                 trainingInstanceVolumeSizeInGB=1024,
                 trainingProjectedColumns=None,
                 trainingChannelName="train",
                 trainingContentType=None,
                 trainingS3DataDistribution="ShardedByS3Key",
                 trainingSparkDataFormat="sagemaker",
                 trainingSparkDataFormatOptions=None,
                 trainingInputMode="File",
                 trainingCompressionCodec=None,
                 trainingMaxRuntimeInSeconds=24*60*60,
                 trainingKmsKeyId=None,
                 modelEnvironmentVariables=None,
                 endpointCreationPolicy=EndpointCreationPolicy.CREATE_ON_CONSTRUCT,
                 sagemakerClient=SageMakerClients.create_sagemaker_client(),
                 region=None,
                 s3Client=SageMakerClients.create_s3_default_client(),
                 stsClient=SageMakerClients.create_sts_default_client(),
                 modelPrependInputRowsToTransformationRows=True,
                 deleteStagingDataAfterTraining=True,
                 namePolicyFactory=RandomNamePolicyFactory(),
                 uid=None,
                 javaObject=None):

        if trainingSparkDataFormatOptions is None:
            trainingSparkDataFormatOptions = {}

        if modelEnvironmentVariables is None:
            modelEnvironmentVariables = {}

        if uid is None:
            uid = Identifiable._randomUID()

        kwargs = locals().copy()
        del kwargs['self']
        super(LinearLearnerRegressor, self).__init__(**kwargs)

        default_params = {
            'predictor_type': 'regressor'
        }

        self._setDefault(**default_params)

    def _get_java_obj(self, **kwargs):
        if 'javaObject' in kwargs and kwargs['javaObject'] is not None:
            return kwargs['javaObject']
        else:
            return self._new_java_obj(
                LinearLearnerRegressor._wrapped_class,
                kwargs['sagemakerRole'],
                kwargs['trainingInstanceType'],
                kwargs['trainingInstanceCount'],
                kwargs['endpointInstanceType'],
                kwargs['endpointInitialInstanceCount'],
                kwargs['requestRowSerializer'],
                kwargs['responseRowDeserializer'],
                kwargs['trainingInputS3DataPath'],
                kwargs['trainingOutputS3DataPath'],
                kwargs['trainingInstanceVolumeSizeInGB'],
                Option(kwargs['trainingProjectedColumns']),
                kwargs['trainingChannelName'],
                Option(kwargs['trainingContentType']),
                kwargs['trainingS3DataDistribution'],
                kwargs['trainingSparkDataFormat'],
                kwargs['trainingSparkDataFormatOptions'],
                kwargs['trainingInputMode'],
                Option(kwargs['trainingCompressionCodec']),
                kwargs['trainingMaxRuntimeInSeconds'],
                Option(kwargs['trainingKmsKeyId']),
                kwargs['modelEnvironmentVariables'],
                kwargs['endpointCreationPolicy'],
                kwargs['sagemakerClient'],
                Option(kwargs['region']),
                kwargs['s3Client'],
                kwargs['stsClient'],
                kwargs['modelPrependInputRowsToTransformationRows'],
                kwargs['deleteStagingDataAfterTraining'],
                kwargs['namePolicyFactory'],
                kwargs['uid']
            )

    @classmethod
    def _from_java(cls, javaObject):
        return LinearLearnerRegressor(sagemakerRole=None, javaObject=javaObject)
