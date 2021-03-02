"""
maml - materials machine learning
"""

__version__ = "2021.3.2"

from .base import BaseDataSource  # noqa
from .base import BaseDescriber  # noqa
from .base import BaseModel, SKLModel, KerasModel, SequentialDescriber  # noqa

__import__('pkg_resources').declare_namespace(__name__)
