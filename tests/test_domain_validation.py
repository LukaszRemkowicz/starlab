"""Invalid API values fail at their domain boundary instead of inside a worker."""

import pytest

from starlab.domain.stars import DetectionSettings


@pytest.mark.parametrize(
    "changes",
    [
        {"min_distance": 2.5},
        {"min_distance": True},
        {"max_stars": 3.5},
        {"max_stars": True},
        {"threshold_sigma": True},
    ],
)
def test_detection_rejects_noninteger_limits_and_boolean_thresholds(changes):
    with pytest.raises(ValueError):
        DetectionSettings(**changes)
