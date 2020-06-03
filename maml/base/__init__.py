# coding: utf-8
# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.

"""
Define abstract base classes.
"""
from ._data import BaseDataSource  # noqa
from ._describer import BaseDescriber, SequentialDescriber  # noqa
from ._describer import DummyDescriber, describer_type  # noqa
from ._model import BaseModel, KerasModel, SKLModel  # noqa
from ._feature_batch import get_feature_batch


__all__ = [
    'BaseDataSource',
    'BaseDescriber',
    'SequentialDescriber',
    'DummyDescriber',
    'BaseModel',
    'KerasModel',
    'SKLModel',
    'get_feature_batch',
    'describer_type'
]
