import numpy as np

from starlab.domain.effects.profiles import transverse_profile


def vectorized_reference(cross, coefficient, denominator):
    """The previous NumPy implementation, kept as the numerical oracle."""
    result = np.maximum(1 - np.abs(cross) * coefficient / denominator, 0) ** 3
    near = np.flatnonzero(cross <= 1.5)
    offsets = np.arange(15) / 14 - 0.5
    shifts = np.arange(-3, 4) / 8
    ratios = coefficient / denominator[near, None]
    result[near] = (np.maximum(1 - np.abs(cross[near, None] + offsets) * ratios, 0) ** 3).mean(1)
    near = near[cross[near] < 1.5]
    shifted = cross[near, None] + shifts
    ratios = coefficient / denominator[near, None]
    point = np.maximum(1 - np.abs(shifted) * ratios, 0) ** 3
    samples = np.maximum(1 - np.abs(shifted[..., None] + offsets) * ratios[..., None], 0) ** 3
    result[near] = np.where(shifted <= 1.5, samples.mean(2), point).mean(1)
    return result


def test_compiled_profile_matches_vectorized_reference():
    rng = np.random.default_rng(7)
    cross = np.r_[rng.uniform(-2, 40, 20000), 1.5, 1.5 - 1e-12, 1.5 + 1e-12, 0.0, 1.875, 1.125]
    denominator = rng.uniform(0.5, 30, len(cross))
    for coefficient in (0.05, 0.4, 2.0):
        np.testing.assert_allclose(
            transverse_profile(cross, coefficient, denominator),
            vectorized_reference(cross, coefficient, denominator),
            rtol=0,
            atol=1e-12,
        )
