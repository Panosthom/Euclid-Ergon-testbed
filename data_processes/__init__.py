"""Researcher data processors.

A processor turns your raw data into train/test tensors (+ optional metadata).
The platform's ``utils.prepare_data`` consumes it by string reference and writes
``data/<dataset>/{train.pt,test.pt,meta.json}`` plus per-client shards under
``data/<dataset>/shards/client_<id>.pt``. The sharding/partitioning engine is
platform-core; only the ``prepare_dataset`` callable is your work product.
"""
