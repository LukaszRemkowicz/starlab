import numpy as np

from starlab.domain.effects.render import LUMA, fit_to_white


def test_in_range_colors_are_unchanged():
    light = np.array([[0.2, 0.5, 1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], np.float32)
    np.testing.assert_array_equal(fit_to_white(light), light)


def test_overbright_color_keeps_hue_and_luminance_on_the_way_to_white():
    orange = np.array([[1.5, 0.75, 0.25]], np.float32)
    fitted = fit_to_white(orange)[0]
    assert fitted.max() == np.float32(1)
    assert fitted @ LUMA == np.float32(orange[0] @ LUMA)
    # Same hue: the offset from gray only shrinks, it does not turn.
    offset, fitted_offset = orange[0] - orange[0] @ LUMA, fitted - fitted @ LUMA
    np.testing.assert_allclose(
        fitted_offset / np.linalg.norm(fitted_offset), offset / np.linalg.norm(offset), atol=1e-6
    )
    assert fitted[0] > fitted[1] > fitted[2]  # still orange, not the clipped yellow (1, 0.75, 0.25)


def test_luminance_at_or_above_one_is_white_and_gray_images_clip():
    np.testing.assert_array_equal(
        fit_to_white(np.array([[4.0, 2.0, 1.0]], np.float32)), [[1, 1, 1]]
    )
    np.testing.assert_array_equal(
        fit_to_white(np.array([[2.0, 2.0, 2.0]], np.float32)), [[1, 1, 1]]
    )
    np.testing.assert_allclose(
        fit_to_white(np.array([[1.7], [0.4]], np.float32)), [[1], [0.4]], rtol=1e-6
    )
