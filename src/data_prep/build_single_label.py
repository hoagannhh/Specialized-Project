"""Compatibility entry point: build all three views with one shared source split.

Bước 0 uses AI-touched labels (generated OR paraphrased) because PADBen Task 2
is included. See prepare_datasets.py for mapping and label-conflict handling.
"""
if __name__ == "__main__":
    try:
        from .prepare_datasets import main
    except ImportError:
        from prepare_datasets import main
    main()
