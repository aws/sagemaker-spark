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

"""Top-level module for sagemaker_pyspark


"""

from .wrapper import SageMakerJavaWrapper, Option
from .IAMRoleResource import IAMRole, IAMRoleFromConfig
from .SageMakerClients import SageMakerClients
from .S3Resources import S3DataPath, S3Resource, S3AutoCreatePath
from .NamePolicy import RandomNamePolicy, RandomNamePolicyFactory, \
                        CustomNamePolicy, CustomNamePolicyFactory, \
                        CustomNamePolicyWithTimeStampSuffix, \
                        CustomNamePolicyWithTimeStampSuffixFactory

from .SageMakerEstimator import EndpointCreationPolicy, SageMakerEstimator, SageMakerEstimatorBase
from .SageMakerModel import SageMakerModel
from .SageMakerResourceCleanup import SageMakerResourceCleanup, CreatedResources


def classpath_jars():
    """Returns a list with the  paths to the required jar files.

    The sagemakerpyspark library is mostly a wrapper of the scala sagemakerspark sdk and it
    depends on a set of jar files to work correctly. This function retrieves the location
    of these jars in the local installation.

    Returns:
        List of absolute paths.
    """
    import pkg_resources
    pkg_dir = __name__
    jars_dir = "/jars/"

    bundled_jars = pkg_resources.resource_listdir(pkg_dir, jars_dir)
    jars = [pkg_resources.resource_filename(pkg_dir, jars_dir + jar) for jar in bundled_jars]

    return jars


__all__ = ['SageMakerJavaWrapper',
           'IAMRole',
           'IAMRoleFromConfig',
           'SageMakerClients',
           'SageMakerModel',
           'S3DataPath',
           'S3AutoCreatePath',
           'S3Resource',
           'EndpointCreationPolicy',
           'SageMakerEstimator',
           'SageMakerEstimatorBase',
           'Option',
           'RandomNamePolicy',
           'RandomNamePolicyFactory',
           'CustomNamePolicy',
           'CustomNamePolicyFactory',
           'CustomNamePolicyWithTimeStampSuffix',
           'CustomNamePolicyWithTimeStampSuffixFactory',
           'classpath_jars',
           'SageMakerResourceCleanup',
           'CreatedResources',
           ]
