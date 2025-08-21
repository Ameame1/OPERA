"""
Utility modules for OPERA framework optimization
These are not Agents but framework-level enhancements
"""

from .query_variant_generator import QueryVariantGenerator
from .placeholder_filler import SmartPlaceholderFiller

__all__ = [
    'QueryVariantGenerator',
    'SmartPlaceholderFiller'
]