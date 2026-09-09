"""IsaacLab-shaped lightweight managers used by benchmark backends."""

from .manager_base import ManagerBase, ManagerTermBase
from .manager_term_cfg import ManagerTermBaseCfg, MeasureTermCfg, TerminationTermCfg
from .measure_manager import MeasureManager
from .termination_manager import TerminationManager

DoneTerm = TerminationTermCfg
MeasureTerm = MeasureTermCfg

__all__ = [
    "DoneTerm",
    "ManagerBase",
    "ManagerTermBase",
    "ManagerTermBaseCfg",
    "MeasureManager",
    "MeasureTerm",
    "MeasureTermCfg",
    "TerminationManager",
    "TerminationTermCfg",
]
