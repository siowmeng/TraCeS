from __future__ import annotations

from omnisafe.algorithms.on_policy import (
    base,
    early_terminated,
    first_order,
    naive_lagrange,
    penalty_function,
    pid_lagrange,
    primal,
    saute,
    second_order,
    simmer,
)

from omnisafe.algorithms.registry import Registry, REGISTRY
# from omnisafe.algorithms.on_policy.naive_lagrange import PPOLag


class TraCeSRegistry(Registry):

    def __init__(self, name: str) -> None:
        """Initialize an instance of :class:`Registry`."""
        super().__init__(name)


TraCeSREGISTRY = TraCeSRegistry('TraCeS')

register = TraCeSREGISTRY.register
get = TraCeSREGISTRY.get

# TraCeSREGISTRY.register(PPOLag)
for algo in [
    *base.__all__,
    *early_terminated.__all__,
    *first_order.__all__,
    *naive_lagrange.__all__,
    *primal.__all__,
    *penalty_function.__all__,
    *pid_lagrange.__all__,
    *saute.__all__,
    *second_order.__all__,
    *simmer.__all__,
]:
    TraCeSREGISTRY.register(REGISTRY.get(algo))

