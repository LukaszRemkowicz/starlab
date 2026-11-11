"""GUI-owned edit history and revision checks for immutable worker snapshots."""

from dataclasses import dataclass, replace
from typing import TypedDict, Unpack

from ..domain.effects.settings import EffectSettings
from ..domain.imaging import ImageFrame, Pixels
from ..domain.sky_mask import SkyMask
from ..domain.stars import DetectionSettings, Star


class StarChanges(TypedDict, total=False):
    """Fields accepted by a manual source edit without erasing untouched measurements."""

    id: int
    x: float
    y: float
    brightness: float
    color: tuple[float, float, float]
    radius: float
    visible: bool
    source_profile: tuple[float, ...]
    soft_weight: float
    source_peak: tuple[float, float, float]


@dataclass(frozen=True)
class EditState:
    """Snapshot source records, controls and their associated detection mask."""

    settings: EffectSettings
    stars: tuple[Star, ...]
    edited: bool
    sky_mask: SkyMask | None = None
    detection_settings: DetectionSettings | None = None


class Session:
    """GUI-thread-owned session. Workers receive immutable snapshots, never this object."""

    def __init__(self) -> None:
        """Initialize an empty image session and bounded undo/redo stacks."""
        self.frame: ImageFrame | None = None
        self.settings = EffectSettings()
        self.stars: tuple[Star, ...] = ()
        self.edited = False
        self.revision = 0
        self.preview: Pixels | None = None
        self.sky_mask: SkyMask | None = None
        self.detection_settings: DetectionSettings | None = None
        self._undo: list[EditState] = []
        self._redo: list[EditState] = []
        self._gesture_start: EditState | None = None

    def _state(self) -> EditState:
        """Snapshot editable records while sharing immutable image-sized data."""
        return EditState(
            self.settings, self.stars, self.edited, self.sky_mask, self.detection_settings
        )

    def _changed(self) -> None:
        """Advance the session revision and invalidate its accepted preview buffer."""
        self.revision += 1
        self.preview = None

    def _remember(self, state: EditState | None = None) -> None:
        """Append a bounded undo snapshot and discard obsolete redo states."""
        self._gesture_start = None
        self._undo.append(state or self._state())
        self._undo = self._undo[-60:]
        self._redo.clear()

    def open(self, frame: ImageFrame) -> None:
        """Replace the source image and clear image-specific edits while retaining controls."""
        self.frame = frame
        self.stars = ()
        self.sky_mask = None
        self.detection_settings = None
        self.edited = False
        self._gesture_start = None
        self._undo.clear()
        self._redo.clear()
        self._changed()

    def set_settings(self, settings: EffectSettings, remember: bool = True) -> None:
        """Apply controls, optionally recording one undo snapshot."""
        if settings == self.settings:
            return
        if self._gesture_start is not None:
            self._remember(self._gesture_start)
        elif remember:
            self._remember()
        self.settings = settings
        self._changed()

    def begin_edit(self) -> None:
        """Hold the pre-gesture state; it enters history only once the gesture changes settings."""
        self._gesture_start = self._state()

    def set_catalog(
        self,
        stars: tuple[Star, ...],
        reset_edits: bool = False,
        sky_mask: SkyMask | None = None,
        detection_settings: DetectionSettings | None = None,
        remember: bool = True,
    ) -> None:
        """Commit detections and their mask, guarding replacement of manual edits."""
        if self.edited and not reset_edits:
            raise ValueError("Reanalysis replaces manual edits; confirm the reset first.")
        if remember:
            self._remember()
        self.stars = stars
        self.sky_mask = sky_mask
        self.detection_settings = detection_settings
        self.edited = False
        self._changed()

    def add_star(self, x: float, y: float) -> None:
        """Place a manually specified source inside the open image and record history."""
        if self.frame is None or not (0 <= x < self.frame.width and 0 <= y < self.frame.height):
            raise ValueError("Place the star inside the image.")
        self._remember()
        self.stars += (
            Star(max((star.id for star in self.stars), default=0) + 1, x, y, 1, (1, 1, 1)),
        )
        self.edited = True
        self._changed()

    def edit_star(self, star_id: int, **changes: Unpack[StarChanges]) -> None:
        """Update a source and invalidate measured profiles when its position changes."""
        current: Star = next(star for star in self.stars if star.id == star_id)
        updated: Star = replace(current, **changes)
        if (updated.x, updated.y) != (current.x, current.y):
            updated = replace(
                updated, source_profile=(), source_peak=(0.0, 0.0, 0.0), soft_weight=0.5
            )
        if self.frame is None or not (
            0 <= updated.x < self.frame.width and 0 <= updated.y < self.frame.height
        ):
            raise ValueError("Star position must be inside the image.")
        self._remember()
        self.stars = tuple(updated if star.id == star_id else star for star in self.stars)
        self.edited = True
        self._changed()

    def _restore(self, source: list[EditState], target: list[EditState]) -> bool:
        """Move one history snapshot between stacks and invalidate stale worker results."""
        if not source:
            return False
        target.append(self._state())
        state: EditState = source.pop()
        self.settings, self.stars, self.edited = state.settings, state.stars, state.edited
        self.sky_mask = state.sky_mask
        self.detection_settings = state.detection_settings
        self._changed()
        return True

    def undo(self) -> bool:
        """Restore the previous committed edit state, if one exists."""
        return self._restore(self._undo, self._redo)

    def redo(self) -> bool:
        """Restore a state removed by Undo, if one exists."""
        return self._restore(self._redo, self._undo)

    def accept_preview(self, revision: int, pixels: Pixels) -> bool:
        """Accept only results calculated for the current session revision."""
        if revision != self.revision:
            return False
        self.preview = pixels
        return True
