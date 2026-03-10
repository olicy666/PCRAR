from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raven3d.pcrar_entity import sample_random_entity
from raven3d.pcrar_rules import DeltaLevel, RuleParams, RuleTemplate, get_rule


def test_count_rule_spreads_three_leaves_for_visibility() -> None:
    rng = np.random.default_rng(0)
    entity = sample_random_entity(rng, leaf_count=1)
    rule = get_rule(RuleTemplate.COUNT)
    params = RuleParams(template=RuleTemplate.COUNT, axis="count", direction=1)

    entity_two = rule.apply(entity, params)
    entity_three = rule.apply(entity_two, params)

    leaves = sorted(entity_three.get_leaves(), key=lambda leaf: (leaf.slot, leaf.id))

    assert len(leaves) == 3
    assert [leaf.slot for leaf in leaves] == [-2, 0, 2]
    assert [leaf.delta_level for leaf in leaves] == [
        DeltaLevel.VFAR,
        DeltaLevel.MID,
        DeltaLevel.VFAR,
    ]
