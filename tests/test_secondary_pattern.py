"""Secondary direction units, rotation and fixed relative amplitudes."""

import pytest

from starlab.domain.effects.profiles import directions
from starlab.domain.effects.settings import EffectParameters


def test_secondary_pattern_uses_tenths_degrees_and_percent_weights():
    rays = directions("secondary", EffectParameters(angle=0))
    assert len(rays) == 84
    assert rays[0] == (40, 0.27)
    assert rays[3] == (160, 0.43)
    assert rays[-1] == (3584, 0.10)
    assert all(0 <= angle < 3600 and 0 < weight <= 1 for angle, weight in rays)


@pytest.mark.parametrize("rotation", [17.5, -90, 180])
def test_secondary_rotation_preserves_order_and_relative_amplitudes(rotation):
    rays = directions("secondary", EffectParameters(angle=rotation))
    base = directions("secondary", EffectParameters(angle=0))
    assert [weight for _, weight in rays] == [weight for _, weight in base]
    assert [angle for angle, _ in rays] == [
        (angle + round(rotation * 10)) % 3600 for angle, _ in base
    ]
