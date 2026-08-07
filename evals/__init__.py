"""Offline evaluation harness for the travel assistant.

Kept out of ``tests/`` on purpose: everything under ``tests/`` runs against
a fake chat model and must stay fast and GPU-free, while these evals load
the real model and measure its behavior. ``tests/test_extraction_eval.py``
covers the scoring logic in here so the harness itself is not the thing
that silently breaks.
"""
