"""Paper-method bridges built on SAEBase's public model/SAE interfaces.

SAR, RSS and Input Localization estimators remain in their original independent repositories.
Supply their modules to the respective functions, or import them using
``load_paper_module`` from an explicit local checkout.
"""

from .localization import (
    LocalizationGrid,
    LocalizationResult,
    collect_localization_grid,
    localization_grid_from_code_store,
    evaluate_localization,
)
from .sources import load_paper_module
from .sar import SARResult, evaluate_sar
from .rss import RSSFeatures, RSSResult, collect_rss_features, evaluate_rss

__all__ = [
    "LocalizationGrid",
    "LocalizationResult",
    "collect_localization_grid",
    "localization_grid_from_code_store",
    "evaluate_localization",
    "load_paper_module",
    "SARResult",
    "evaluate_sar",
    "RSSFeatures",
    "RSSResult",
    "collect_rss_features",
    "evaluate_rss",
]
