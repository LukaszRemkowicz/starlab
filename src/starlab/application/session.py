from dataclasses import dataclass, replace

from ..domain.effects.settings import EffectSettings
from ..domain.imaging import ImageFrame, Pixels
from ..domain.sky_mask import SkyMask
from ..domain.stars import DetectionSettings, Star


@dataclass(frozen=True)
class EditState:
    settings: EffectSettings
    stars: tuple[Star, ...]
    edited: bool
    sky_mask: SkyMask | None = None
    detection_settings: DetectionSettings | None = None


class Session:
    """GUI-thread-owned session. Workers receive immutable snapshots, never this object."""

    def __init__(self) -> None:
        self.frame: ImageFrame | None = None
        self.settings = EffectSettings()
        self.stars: tuple[Star, ...] = ()
        self.edited = False
        self.revision = 0
        self.preview: Pixels | None = None
        self.sky_mask = None
        self.detection_settings = None
        self._undo: list[EditState] = []
        self._redo: list[EditState] = []

    def _state(self) -> EditState:
        return EditState(
            self.settings, self.stars, self.edited, self.sky_mask, self.detection_settings
        )

    def _changed(self) -> None:
        self.revision += 1
        self.preview = None

    def _remember(self) -> None:
        self._undo.append(self._state())
        self._undo = self._undo[-60:]
        self._redo.clear()

    def open(self, frame: ImageFrame) -> None:
        self.frame = frame
        self.stars = ()
        self.sky_mask = None
        self.detection_settings = None
        self.edited = False
        self._undo.clear()
        self._redo.clear()
        self._changed()

    def set_settings(self, settings: EffectSettings, remember: bool = True) -> None:
        if settings == self.settings:
            return
        if remember:
            self._remember()
        self.settings = settings
        self._changed()

    def begin_edit(self) -> None:
        self._remember()

    def set_catalog(
        self,
        stars: tuple[Star, ...],
        reset_edits: bool = False,
        sky_mask=None,
        detection_settings=None,
        remember: bool = True,
    ) -> None:
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
        if self.frame is None or not (0 <= x < self.frame.width and 0 <= y < self.frame.height):
            raise ValueError("Place the star inside the image.")
        self._remember()
        self.stars += (Star(max((s.id for s in self.stars), default=0) + 1, x, y, 1, (1, 1, 1)),)
        self.edited = True
        self._changed()

    def edit_star(self, star_id: int, **changes) -> None:
        current = next(s for s in self.stars if s.id == star_id)
        updated = replace(current, **changes)
        if (updated.x, updated.y) != (current.x, current.y):
            updated = replace(
                updated, source_profile=(), source_peak=(0.0, 0.0, 0.0), soft_weight=0.5
            )
        if self.frame is None or not (
            0 <= updated.x < self.frame.width and 0 <= updated.y < self.frame.height
        ):
            raise ValueError("Star position must be inside the image.")
        self._remember()
        self.stars = tuple(updated if s.id == star_id else s for s in self.stars)
        self.edited = True
        self._changed()

    def _restore(self, source: list[EditState], target: list[EditState]) -> bool:
        if not source:
            return False
        target.append(self._state())
        state = source.pop()
        self.settings, self.stars, self.edited = state.settings, state.stars, state.edited
        self.sky_mask = state.sky_mask
        self.detection_settings = state.detection_settings
        self._changed()
        return True

    def undo(self) -> bool:
        return self._restore(self._undo, self._redo)

    def redo(self) -> bool:
        return self._restore(self._redo, self._undo)

    def accept_preview(self, revision: int, pixels: Pixels) -> bool:
        if revision != self.revision:
            return False
        self.preview = pixels
        return True
