from dataclasses import replace

import numpy as np


def session():
    from starlab.application.session import Session
    from starlab.domain.imaging import ImageFrame

    s = Session()
    s.open(ImageFrame(np.zeros((32, 32, 3), np.float32)))
    return s


def test_settings_history_preserves_catalog_and_rejects_stale_render():
    from starlab.domain.stars import Star

    s = session()
    s.set_catalog((Star(1, 12, 13, 1, (1, 1, 1), 2),))
    old_revision = s.revision
    original = s.settings
    new = replace(original, primary=replace(original.primary, angle=45))
    s.set_settings(new)
    assert len(s.stars) == 1
    assert not s.accept_preview(old_revision, np.zeros((32, 32, 3), np.float32))
    assert s.undo()
    assert s.settings == original
    assert s.redo()
    assert s.settings == new


def test_manual_edits_can_be_undone_and_reanalysis_requires_explicit_reset():
    import pytest

    s = session()
    s.add_star(5.5, 6.5)
    star = s.stars[0]
    s.edit_star(star.id, x=10, visible=False)
    assert not s.stars[0].visible
    s.undo()
    assert s.stars[0].visible
    assert s.stars[0].x == 5.5
    with pytest.raises(ValueError):
        s.set_catalog(())
    s.set_catalog((), reset_edits=True)
    assert s.stars == ()


def test_open_resets_stars_and_undo_but_keeps_settings():
    from starlab.domain.imaging import ImageFrame

    s = session()
    s.add_star(3, 4)
    settings = s.settings
    s.open(ImageFrame(np.zeros((10, 10, 1), np.float32)))
    assert s.stars == ()
    assert not s.undo()
    assert s.settings == settings


def test_moving_source_discards_old_measurement_and_undo_restores_it():
    from starlab.domain.stars import Star

    s = session()
    profile = (3, 0.02, 0.02, 0.02, 0, 0, 0, 0, 0, 0)
    source = Star(
        1,
        12,
        13,
        1,
        (1, 1, 1),
        source_profile=profile,
        source_peak=(0.2, 0.4, 0.6),
        soft_weight=0.1,
    )
    s.set_catalog((source,))
    s.edit_star(1, x=12, visible=False)
    assert s.stars[0].source_profile == profile
    s.edit_star(1, x=14)
    assert s.stars[0].source_profile == ()
    assert s.stars[0].source_peak == (0, 0, 0)
    assert s.stars[0].soft_weight == 0.5
    assert s.undo()
    assert s.stars[0].source_profile == profile
    assert s.stars[0].source_peak == source.source_peak
    assert s.stars[0].soft_weight == 0.1


def test_slider_gesture_records_one_state_and_noop_click_keeps_history():
    s = session()
    original = s.settings
    first = replace(original, primary=replace(original.primary, angle=45))
    s.set_settings(first)
    assert s.undo()
    for _ in range(70):  # clicks without movement must not evict history or clear redo
        s.begin_edit()
    assert s.redo()
    assert s.settings == first
    s.begin_edit()
    for angle in (50, 55, 60):
        s.set_settings(replace(first, primary=replace(first.primary, angle=angle)), remember=False)
    assert s.undo()
    assert s.settings == first
    assert s.undo()
    assert s.settings == original
