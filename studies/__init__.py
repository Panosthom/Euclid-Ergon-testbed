"""Researcher experiment-spec factories.

This package is named ``studies`` (not ``experiments``) on purpose: the platform
already ships a top-level ``experiments`` package (``experiments.api``,
``experiments.submission``, ``experiments.run_submitted``). A second top-level
``experiments`` package in this template would shadow the platform's and break
``import experiments.api``. Keep researcher specs here until/unless the platform
is namespaced (e.g. ``fedplatform.experiments``), at which point this can be
renamed freely.
"""
