"""Preprocessing engine (PHASE4_PLAN.md).

``spec``, ``catalog`` and ``validate`` are standard-library only and safe for the API to
import. ``runtime`` holds the algorithms (embedded into generated code); ``base``, ``ops``,
``executor`` and ``report`` run on pandas/scikit-learn in workers and tests.
"""
