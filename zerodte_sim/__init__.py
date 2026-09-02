"""Monte Carlo simulator for short 0DTE vertical spreads and roll policies."""

from .config import AccountConfig, CostModel, FilterConfig, StrategyConfig
from .engine import DayResult, PathResult, simulate_day, simulate_path
from .experiment import Experiment, Variant, default_variants
from .filter_study import FilterSpec, calibrate_threshold, run_filter_study
from .market import MarketConfig, MarketSimulator
from .metrics import Summary, summarise

__all__ = [
    "AccountConfig",
    "CostModel",
    "FilterConfig",
    "StrategyConfig",
    "DayResult",
    "PathResult",
    "simulate_day",
    "simulate_path",
    "Experiment",
    "Variant",
    "default_variants",
    "FilterSpec",
    "calibrate_threshold",
    "run_filter_study",
    "MarketConfig",
    "MarketSimulator",
    "Summary",
    "summarise",
]
