import itertools
from types import MappingProxyType

# On-Policy Safe
from ubcrl.algorithms.on_policy import PPOLagLearnedBC, PPOLagLearnedH

ALGORITHMS = {
    'on-policy': tuple(on_policy.__all__),
}

ALGORITHM2TYPE = {
    algo: algo_type for algo_type, algorithms in ALGORITHMS.items() for algo in algorithms
}

__all__ = ALGORITHMS['all'] = tuple(itertools.chain.from_iterable(ALGORITHMS.values()))

assert len(ALGORITHM2TYPE) == len(__all__), 'Duplicate algorithm names found.'

ALGORITHMS = MappingProxyType(ALGORITHMS)  # make this immutable
ALGORITHM2TYPE = MappingProxyType(ALGORITHM2TYPE)  # make this immutable

del itertools, MappingProxyType
