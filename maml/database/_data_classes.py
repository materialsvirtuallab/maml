"""
dataclasses for documents in the database
"""


from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional


@dataclass
class DataInfo:
    """
    data contained in a data_info doc
    """
    data_name: str
    input_type: str
    target_unit: str
    target_type: str = 'intensive'
    task_type: str = 'regression'


@dataclass
class DataSplit:
    """
    data contained in a data_split doc
    """
    data_name: str
    split_name: str
    unique_id: str
    train_ids: List
    val_id: List
    test_id: List


@dataclass
class MaterialData:
    """
    The data class that stores one document of material data.
    required properties include data_name. Extra necessary fields
    will include the input, target and id. For example
    for
    """
    data_name: str
    input: Any
    target: Any
    id: Union[str, int]


@dataclass
class DescriberData:
    """
    The describer data 
    """
    describer_name: str
    describer_params: Dict
    state: Any


@dataclass
class LearningModelData:
    """
    learning model information, will be stored in a dict
    """
    learning_model_name: str
    model_type: str
    model_params: Dict
    state: Any


@dataclass
class ModelData:
    """
    The model information. ModelData will contain
    DescriberData and LearningModelData
    """
    model_name: str
    describer: DescriberData
    model: LearningModelData
    descriptor_name: Optional[str] = None


@dataclass
class ModelResults:
    """
    Used to save model results
    """
    model_name: str
    metric_name: str
    test_data_name: str
    test_split_name: str
    metric_value: float
    predicted_tests: List
    true_tests: List
