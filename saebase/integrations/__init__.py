"""Paper-method bridges built on SAEBase's public model/SAE interfaces.

SAR and RSS estimators remain in their original independent repositories.
Supply their modules to the respective functions, or import them using
``load_paper_module`` from an explicit local checkout.
"""

from .sources import load_paper_module
from .sar import SARResult, evaluate_sar
from .rss import RSSFeatures, RSSResult, collect_rss_features, evaluate_rss

__all__ = [
    "load_paper_module", "SARResult", "evaluate_sar",
    "RSSFeatures", "RSSResult", "collect_rss_features", "evaluate_rss",
]
