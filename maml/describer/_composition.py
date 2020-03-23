"""
Compositional describers
"""

from matminer.featurizers.composition import ElementProperty as MatminerElementProperty  # noqa
from ._matminer_wrapper import wrap_matminer_describer

ElementProperty = wrap_matminer_describer("ElementProperty", MatminerElementProperty)
