"""Contains all the enums"""
from enum import Enum


class DatasetSplitType(Enum):
    """Enum for dataset type, i.e. train/test/validation"""

    TRAIN = "train"
    TEST = "test"
    VALIDATION = "validation"


class EnvironmentType(Enum):
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    NUMPY = "numpy"
