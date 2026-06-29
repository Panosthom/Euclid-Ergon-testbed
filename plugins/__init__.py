"""Researcher plugins for the FL Edge platform.

Each submodule subclasses a platform ABC (or provides the functional client
factory) and self-registers at import time. Bootstrap them all via the top-level
``register`` module; do not import platform internals here.
"""
