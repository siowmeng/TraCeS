from __future__ import annotations

import os
from typing import Any

from omnisafe.common.experiment_grid import ExperimentGrid
from omnisafe.common.statistics_tools import StatisticsTools
from omnisafe.utils.tools import (
    load_yaml,
    recursive_check_config,
)

from ubcrl.algorithms import ALGORITHM2TYPE
from ubcrl.evaluator import UBCRLEvaluator


# pylint: disable-next=too-many-instance-attributes
class UBCRLExperimentGrid(ExperimentGrid):

    _evaluator: UBCRLEvaluator

    # pylint: disable-next=too-many-locals

    def check_variant_vaild(self, variant: dict[str, Any]) -> None:
        """Check if the variant is valid.

        Args:
            variant (dict[str, Any]): Experiment variant to be checked.
        """
        path = os.path.dirname(os.path.abspath(__file__))
        algo_type = ALGORITHM2TYPE.get(variant['algo'], '')
        cfg_path = os.path.join(path, '..', 'configs', algo_type, f"{variant['algo']}.yaml")
        default_config = load_yaml(cfg_path)['defaults']
        recursive_check_config(variant, default_config, exclude_keys=('algo', 'env_id'))

    def _init_statistical_tools(self) -> None:
        """Initialize statistical tools."""
        self._statistical_tools = StatisticsTools()
        self._evaluator = UBCRLEvaluator()

