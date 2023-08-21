"""Contains all the enums"""
from enum import Enum


class SetType(Enum):
    """Enum for dataset type, i.e. train/test/validation"""

    TRAIN = "train"
    TEST = "test"
    VALIDATION = "validation"
