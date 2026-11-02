"""Independent manifest preparation must not pretend to run native checks."""

import pytest

from scripts.workplan.wp023_orchestrator import prepare_inventory


def test_invalid_run_id_rejected(tmp_path):
    with pytest.raises(ValueError, match="run ID"):
        prepare_inventory("../escape", tmp_path)


def test_missing_repository_inventory_fails_closed(tmp_path):
    with pytest.raises((ModuleNotFoundError, FileNotFoundError, ValueError)):
        prepare_inventory("run01", tmp_path)
