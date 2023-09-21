import logging
import logging.config
import os
from contextvars import ContextVar
from typing import Any, Optional

import yaml

is_init = False
COMMON_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def init(
    default_level: int = logging.INFO,
) -> None:
    """Sets up logging with config file, if present or basic config otherwise"""
    # basicConfig is no-op if handlers already exists for root logger
    _set_basic_logging_cfg(default_level)


def _set_basic_logging_cfg(level: int) -> None:
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    logging.basicConfig(level=level, format=COMMON_FORMAT)


def get_logger(logger_name: Optional[str] = None) -> logging.Logger:
    init()
    if logger_name:
        return logging.getLogger(logger_name)
    return logging.getLogger()


logger = get_logger()
