"""Single bootstrap module: importing it fires every plugin's ``register_*`` call.

Any experiment entrypoint MUST import this module before submitting a spec, so
that all components are discoverable by their string names. Importing each plugin
submodule executes the module-level ``register_model`` / ``register_aggregator`` /
``register_client_fn`` / ``register_dataset_loader`` calls.

    import register  # noqa: F401  -> all plugins now registered

Add one import line per new plugin module you create under ``plugins/``.
"""

from __future__ import annotations

# Models
import plugins.models.example_model  # noqa: F401
import plugins.models.tabular_mlp  # noqa: F401

# Aggregators
import plugins.aggregators.example_aggregator  # noqa: F401

# Clients
import plugins.clients.example_client  # noqa: F401

# Datasets
import plugins.datasets.example_dataset  # noqa: F401


def registered() -> dict:
    """Return the registry contents, for a quick `python register.py` sanity check."""
    from registry import list_registered_components

    return {k: list(v) for k, v in list_registered_components().items()}


if __name__ == "__main__":
    import json

    print(json.dumps(registered(), indent=2))
