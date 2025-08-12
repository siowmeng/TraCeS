from __future__ import annotations

from omnisafe.algorithms.registry import Registry
from omnisafe.algorithms.on_policy.naive_lagrange import PPOLag

class UBCRLRegistry(Registry):

    def __init__(self, name: str) -> None:
        """Initialize an instance of :class:`Registry`."""
        super().__init__(name)


UBCRLREGISTRY = UBCRLRegistry('UBCRL')


register = UBCRLREGISTRY.register
get = UBCRLREGISTRY.get

UBCRLREGISTRY.register(PPOLag)
