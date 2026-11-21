"""Compatibility CLI delegating saved SAR pattern searches to formal Capture."""

from tasks.capture.pair_search import main, scan_run, search_patterns

__all__ = ["main", "scan_run", "search_patterns"]

if __name__ == "__main__":
    main()
