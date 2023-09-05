"""
    Defines all the DataModels related to dataset
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class DatasetObject:
    """
    Model which stores the information about data element
    """

    path: str
    name: str
    label: Optional[Any] = None
    mask_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
