import logging
import logging.config
import os
from contextvars import ContextVar
from typing import Any, Optional

import yaml

is_init = False
COMMON_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def init(
    default_path: str = "logging.yaml",
    default_level: int = logging.INFO,
    env_key: str = "LOG_CFG",
) -> None:
    """Sets up logging with config file, if present or basic config otherwise"""
    # basicConfig is no-op if handlers already exists for root logger
    global is_init  # pylint: disable=global-statement
    if is_init:
        return
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, "rt", encoding="utf-8") as f:
            try:
                config = yaml.safe_load(f.read())
                logging.config.dictConfig(config)
            except Exception as e:
                # pylint: disable-next=no-print
                print(f"Error in log config {path}: {e}. Using default configs")
                _set_basic_logging_cfg(default_level)

    else:
        # pylint: disable-next=no-print
        print("Failed to load log config file. Using default configs")
        _set_basic_logging_cfg(default_level)
    is_init = True


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

# Create our context var - this will be filled at the start of request
# processing in fastapi methods
ctx_user: ContextVar[str] = ContextVar("user", default="Unknown")


class ContextFilter(logging.Filter):
    """
    This is a filter which injects contextual information into the log.
    """

    def filter(self, record: Any) -> bool:  # pylint: disable=no-any
        record.user = ctx_user.get()
        return True
