"""
qSOFA (Quick Sequential Organ Failure Assessment) Model Package.

This package provides a rule-based implementation of the qSOFA scoring
system from the Sepsis-3 guidelines for sepsis prediction.
"""

from .qsofa_model import QSOFAModel

__all__ = ['QSOFAModel']
