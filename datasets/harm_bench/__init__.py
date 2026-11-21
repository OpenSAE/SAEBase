"""HarmBench source preparation and fixed sample selection."""

from functools import partial
from saebase.datasets.public_sources import prepare_public_dataset, select_public_samples

prepare_for_model = partial(prepare_public_dataset, "HarmBench")
select_samples = partial(select_public_samples, "HarmBench")
SUPPORTED_TASKS = frozenset({"capture", "intervene", "generate", "precheck"})
