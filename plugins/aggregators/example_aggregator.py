"""Example aggregator plugins — swap the federated **aggregation algorithm**.

The platform's server runs FedAvg by default. To use a different strategy, point
your spec's ``aggregator`` at a ``flwr.server.strategy.Strategy`` **subclass** via
the ``source`` kwarg::

    aggregator=ComponentRef("fedprox",
                            {"source": "plugins.aggregators.example_aggregator:FedProxStrategy"})

The server resolves ``--aggregator-source`` to your class and wraps it with the
testbed harness (global-model saving + centralized global-test evaluation + CSV
logging) automatically — you only provide the aggregation logic.

Contract: your class must accept FedAvg-style constructor kwargs (``fraction_fit``,
``min_fit_clients``, ``min_available_clients``, ``on_fit_config_fn``, ...) and
**default any algorithm-specific params**, because the server constructs it with the
standard kwargs. Pass-through ``**kwargs`` to ``super().__init__`` and add your own.
Works in both simulation and on the Pis.
"""

from __future__ import annotations

from typing import Any

from flwr.server.strategy import FedAdam, FedProx


class FedProxStrategy(FedProx):
    """FedProx with a sensible default proximal term (μ=0.1)."""

    def __init__(self, *, proximal_mu: float = 0.1, **kwargs: Any) -> None:
        super().__init__(proximal_mu=proximal_mu, **kwargs)


class FedAdamStrategy(FedAdam):
    """Server-side adaptive optimization (FedAdam) with default learning rates."""

    def __init__(self, *, eta: float = 1e-2, eta_l: float = 1e-1, tau: float = 1e-3, **kwargs: Any) -> None:
        super().__init__(eta=eta, eta_l=eta_l, tau=tau, **kwargs)


# To write a fully custom rule, subclass FedAvg and override aggregate_fit:
#
#     from flwr.server.strategy import FedAvg
#     class MyStrategy(FedAvg):
#         def aggregate_fit(self, server_round, results, failures):
#             # your weighting / robust aggregation here
#             return super().aggregate_fit(server_round, results, failures)
#
# then reference it via aggregator=ComponentRef("mine",
#     {"source": "plugins.aggregators.example_aggregator:MyStrategy"}).
