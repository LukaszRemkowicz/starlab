"""Tiled recovered analysis with explicitly approximate, accelerated seeds.

The original tests every above-threshold pixel. Here a 3x3 local-maximum stage
finds peaks, then recovered analysis qualifies a peak and its 5x5 neighborhood.
This removes redundant work on broad backgrounds but can miss non-maximal,
crowded or asymmetric sources. It is not a pixel-exact original scan.
Normalization and the output catalog are global, never ranked within tiles.
"""

import logging
import os
import time
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import closing
from functools import partial
from threading import Event
from typing import cast

import numpy as np
from scipy import ndimage

from ..array_types import (
    BoolArray,
    Bounds,
    ByteArray,
    CandidateRecord,
    Diagnostics,
    DoubleArray,
    FloatArray,
)
from ..imaging import ImageFrame, check_cancel
from ..sky_mask import SkyMask
from .buffers import anonymous_array
from .candidate import Image, candidate, covered, merge_candidate
from .catalog import join_catalogs
from .constants import PREFERRED_SEED_OFFSETS
from .numeric import normalize_catalog
from .peaks import peak_positions
from .proxy import coarse_codes, coarse_step, decode, full_codes, full_proxy, maximum_luminance
from .records import IndexArray, PreparedTile, ReferenceAttempt, TileKey, TileResult

TILE = 1024
HALO = 272  # Covers native 266-pixel rays and the largest disk/ring support.
logger = logging.getLogger(__name__)
COMPILED = True


def _tile_region(
    width: int, height: int, left: int, top: int, sky_mask: SkyMask | None, source_scale: int
) -> tuple[Bounds, Bounds, BoolArray | None]:
    """Return tile core, clipped halo bounds, and allowed pixels at source scale."""
    core: Bounds = (left, top, min(width, left + TILE), min(height, top + TILE))
    bounds: Bounds = (
        max(0, left - HALO),
        max(0, top - HALO),
        min(width, core[2] + HALO),
        min(height, core[3] + HALO),
    )
    allowed: None | BoolArray = (
        None
        if sky_mask is None
        else sky_mask.region(
            cast(Bounds, tuple(coordinate * source_scale for coordinate in bounds)),
            sampling=source_scale,
        )
    )
    return core, bounds, allowed


def peak_positions_reference(
    image: Image, threshold: float, core: Bounds, coarse: bool = False
) -> Iterator[tuple[int, int]]:
    """Yield global pixel-index component centers using the full SciPy reference screens."""
    intensity: FloatArray = image.rgb.max(axis=2)
    allowed: BoolArray | None = image.allowed
    maxima: FloatArray = ndimage.maximum_filter(
        image.luminance if allowed is None else np.where(allowed, image.luminance, -np.inf),
        size=3,
        mode="nearest",
    )
    floor: FloatArray = ndimage.minimum_filter(
        image.luminance if allowed is None else np.where(allowed, image.luminance, np.inf),
        size=129,
        mode="nearest",
    )
    support: FloatArray = ndimage.uniform_filter(
        ((intensity > threshold * 0.05) & (True if allowed is None else allowed)).astype(
            np.float32
        ),
        size=3,
        mode="constant",
    )
    selected: BoolArray = (
        (image.luminance == maxima)
        & (intensity >= threshold)
        & (image.luminance - floor >= 0.13)
        & (support >= 2.5 / 9)
    )
    if allowed is not None:
        selected &= allowed
    if coarse:
        # This pass keeps only size>4. Native98440 requires center maximum>=.75
        # and normalized inner-disk score>=.67. The3x3 mean is a deliberately
        # approximate early screen for that broad bright support, not a rank.
        selected &= (image.luminance >= 0.75) & (
            ndimage.uniform_filter(image.luminance, size=3, mode="nearest") >= 0.6
        )
        contrast_allowed: BoolArray | None = None
    else:
        # Approximate early screen inspired by the recovered mean-std shape
        # score. A local peak on a textured bright background can satisfy the
        # distant minimum test while having no compact source contrast.
        values: FloatArray = (
            image.luminance if allowed is None else np.where(allowed, image.luminance, 0)
        )
        mean: FloatArray = ndimage.uniform_filter(values, size=7, mode="nearest")
        square: FloatArray = ndimage.uniform_filter(values**2, size=7, mode="nearest")
        if allowed is not None:
            count: FloatArray = ndimage.uniform_filter(
                allowed.astype(np.float32), size=7, mode="nearest"
            )
            mean = np.divide(mean, count, out=np.zeros_like(mean), where=count > 0)
            square = np.divide(square, count, out=np.zeros_like(square), where=count > 0)
        stddev: FloatArray = np.sqrt(np.maximum(square - mean**2, 0))
        contrast_allowed = (image.luminance >= 0.75) | (image.luminance - mean + stddev >= 0.08)
    labels: IndexArray
    labels, _ = ndimage.label(selected)
    for region in ndimage.find_objects(labels):
        if region is None:
            continue
        y: int = (region[0].start + region[0].stop - 1) // 2
        x: int = (region[1].start + region[1].stop - 1) // 2
        if contrast_allowed is not None and not contrast_allowed[y, x]:
            continue
        x += image.origin_x
        y += image.origin_y
        if core[0] <= x < core[2] and core[1] <= y < core[3]:
            yield x, y


def _reference_attempt(
    image: Image,
    x: int,
    y: int,
    threshold: float,
    coarse: bool,
    excluded: Sequence[CandidateRecord],
    visited: set[tuple[int, int]],
) -> ReferenceAttempt | None:
    """Record one fresh above-threshold seed, including rejected candidates."""
    if image.rgb[y - image.origin_y, x - image.origin_x].max() < threshold:
        return None
    visited.add((x, y))
    result: CandidateRecord | None = candidate(image, x + 0.5, y + 0.5, coarse, excluded)
    return y, x, result


def _reference_neighbors(
    image: Image,
    peak_x: int,
    peak_y: int,
    width: int,
    height: int,
    threshold: float,
    neighbor_radius: int,
    coarse: bool,
    excluded: Sequence[CandidateRecord],
    visited: set[tuple[int, int]],
) -> list[ReferenceAttempt]:
    """Evaluate unvisited neighborhood seeds in native row order."""
    attempts: list[tuple[int, int, CandidateRecord | None]] = []
    for y in range(max(0, peak_y - neighbor_radius), min(height, peak_y + neighbor_radius + 1)):
        for x in range(max(0, peak_x - neighbor_radius), min(width, peak_x + neighbor_radius + 1)):
            if (x, y) in visited or covered(x + 0.5, y + 0.5, excluded):
                continue
            attempt: tuple[int, int, CandidateRecord | None] | None = _reference_attempt(
                image, x, y, threshold, coarse, excluded, visited
            )
            if attempt is not None:
                attempts.append(attempt)
    return attempts


def _reference_attempts(
    image: Image,
    peak_x: int,
    peak_y: int,
    width: int,
    height: int,
    threshold: float,
    neighbor_radius: int,
    coarse: bool,
    excluded: Sequence[CandidateRecord],
    visited: set[tuple[int, int]],
) -> list[ReferenceAttempt]:
    """Try the preferred basin seeds, then expand only after an accepted candidate."""
    accepted: bool = False
    # A maximum may itself fail the original shape predicate;
    # four adjacent seeds retain the nearby qualifying basin.
    attempts: list[tuple[int, int, CandidateRecord | None]] = []
    for dx, dy in PREFERRED_SEED_OFFSETS:
        x, y = peak_x + dx, peak_y + dy
        if not (0 <= x < width and 0 <= y < height):
            continue
        if (x, y) in visited:
            continue
        attempt: tuple[int, int, CandidateRecord | None] | None = _reference_attempt(
            image, x, y, threshold, coarse, excluded, visited
        )
        if attempt is None:
            continue
        attempts.append(attempt)
        if attempt[2] is not None:
            accepted = True
            break
    if accepted:
        attempts.extend(
            _reference_neighbors(
                image,
                peak_x,
                peak_y,
                width,
                height,
                threshold,
                neighbor_radius,
                coarse,
                excluded,
                visited,
            )
        )
    return attempts


def _scan_reference_tile(
    image: Image,
    core: Bounds,
    width: int,
    height: int,
    threshold: float,
    neighbor_radius: int,
    coarse: bool,
    excluded: Sequence[CandidateRecord],
    visited: set[tuple[int, int]],
    cancel: Event | None,
) -> tuple[list[CandidateRecord], int, int]:
    """Qualify a reference tile and merge accepted attempts in native seed order."""
    local_rows: list[CandidateRecord] = []
    maximum_size: float = 0.0
    peaks: int = 0
    seeds: int = 0
    for peak_x, peak_y in peak_positions_reference(image, threshold, core, coarse):
        peaks += 1
        check_cancel(cancel)
        if covered(peak_x + 0.5, peak_y + 0.5, excluded):
            continue
        attempts: list[tuple[int, int, CandidateRecord | None]] = _reference_attempts(
            image,
            peak_x,
            peak_y,
            width,
            height,
            threshold,
            neighbor_radius,
            coarse,
            excluded,
            visited,
        )
        seeds += len(attempts)
        # Native local merging scans seed rows left-to-right.
        for _, _, result in sorted(attempts, key=lambda row: row[:2]):
            if result is not None:
                maximum_size = merge_candidate(local_rows, result, maximum_size)
    return local_rows, peaks, seeds


def scan_tiled(
    width: int,
    height: int,
    read_tile: Callable[[Bounds], FloatArray],
    threshold: float,
    neighbor_radius: int,
    cancel: Event | None,
    coarse: bool = False,
    excluded: Sequence[CandidateRecord] = (),
    sky_mask: SkyMask | None = None,
    source_scale: int = 1,
) -> tuple[list[CandidateRecord], Diagnostics]:
    """Scan all cores with halo sampling, native merges, and a shared global seed-visited map."""
    if COMPILED:
        return scan_compiled(
            width,
            height,
            read_tile,
            threshold,
            neighbor_radius,
            cancel,
            coarse,
            excluded,
            sky_mask,
            source_scale,
        )
    rows: list[CandidateRecord] = []
    visited: set[tuple[int, int]] = set()
    peaks: int = 0
    seeds: int = 0
    for top in range(0, height, TILE):
        for left in range(0, width, TILE):
            tile_started: float = time.monotonic()
            check_cancel(cancel)
            core, bounds, allowed = _tile_region(width, height, left, top, sky_mask, source_scale)
            if allowed is not None and not allowed.any():
                continue
            image: Image = Image(read_tile(bounds), origin=(bounds[0], bounds[1]), allowed=allowed)
            try:
                local_rows, count_peaks, count_seeds = _scan_reference_tile(
                    image,
                    core,
                    width,
                    height,
                    threshold,
                    neighbor_radius,
                    coarse,
                    excluded,
                    visited,
                    cancel,
                )
                peaks += count_peaks
                seeds += count_seeds
            finally:
                image.clear_cache()
            join_catalogs(rows, local_rows)
            logger.info(
                "Detector %s tile (%d,%d): %d peaks, %d seeds, %d sources; %.2fs",
                "coarse" if coarse else "full",
                left,
                top,
                peaks,
                seeds,
                len(rows),
                time.monotonic() - tile_started,
            )
    return rows, dict(peaks=peaks, candidate_seeds=seeds)


WORKERS = min(16, os.cpu_count() or 1)
# Each in-flight halo tile holds ~40 MB of float32 proxy/luminance: one per worker bounds memory.
IN_FLIGHT = WORKERS


def _prepare_tile(
    row: int,
    column: int,
    *,
    width: int,
    height: int,
    read_tile: Callable[[Bounds], FloatArray],
    threshold: float,
    coarse: bool,
    token: Event,
    sky_mask: SkyMask | None,
    source_scale: int,
) -> PreparedTile:
    """Read one halo tile and locate peaks without writing shared visitation state."""
    started: float = time.monotonic()
    check_cancel(token)
    left, top = column * TILE, row * TILE
    core, bounds, allowed = _tile_region(width, height, left, top, sky_mask, source_scale)
    if allowed is not None and not allowed.any():
        return None, [], np.zeros(2)
    image: Image = Image(read_tile(bounds), origin=(bounds[0], bounds[1]), allowed=allowed)
    converted: float = time.monotonic()
    try:
        positions: list[tuple[int, int]] = list(peak_positions(image, threshold, core, coarse))
        return image, positions, np.array([converted - started, time.monotonic() - converted])
    except BaseException:
        image.clear_cache()
        raise


def _qualify_tile(
    value: PreparedTile,
    *,
    threshold: float,
    neighbor_radius: int,
    coarse: bool,
    excluded: Sequence[CandidateRecord],
    visited: BoolArray,
    token: Event,
) -> TileResult:
    """Qualify a prepared tile and release its caches even on cancellation."""
    from .compiled import scan_tile

    image, positions, timing = value
    if image is None:
        return [], 0, 0, np.zeros(3)
    started: float = time.monotonic()
    try:
        local, count = scan_tile(
            image, positions, threshold, neighbor_radius, coarse, excluded, visited, token
        )
        return (
            local,
            len(positions),
            count,
            np.array([timing[0], timing[1], time.monotonic() - started]),
        )
    finally:
        image.clear_cache()


def _tile_ready(row: int, column: int, columns: int, completed: set[TileKey]) -> bool:
    """Preserve adjacent visitation writes before a tile starts qualification."""
    return (column == 0 or (row, column - 1) in completed) and (
        row == 0 or (row - 1, min(column + 1, columns - 1)) in completed
    )


def _wait_tiles(
    preparing: dict[Future[PreparedTile], TileKey],
    qualifying: dict[Future[TileResult], TileKey],
    prepared: dict[TileKey, PreparedTile],
    results: dict[TileKey, TileResult],
    completed: set[TileKey],
) -> None:
    """Collect completed preparation/qualification work without changing join order."""
    # wait only observes futures; membership below identifies their concrete result layout.
    pending: Iterable[
        Future[
            tuple[Image | None, list[tuple[int, int]], DoubleArray]
            | tuple[list[CandidateRecord], int, int, DoubleArray]
        ]
    ] = cast(Iterable[Future[PreparedTile | TileResult]], (*preparing, *qualifying))
    finished, _ = wait(pending, return_when=FIRST_COMPLETED)
    for future in finished:
        if future in preparing:
            prepared_future: Future[tuple[Image | None, list[tuple[int, int]], DoubleArray]] = cast(
                Future[PreparedTile], future
            )
            prepared[preparing.pop(prepared_future)] = prepared_future.result()
        else:
            qualified_future: Future[tuple[list[CandidateRecord], int, int, DoubleArray]] = cast(
                Future[TileResult], future
            )
            key: tuple[int, int] = qualifying.pop(qualified_future)
            results[key] = qualified_future.result()
            completed.add(key)


def _cancel_tiles(
    token: Event,
    preparing: dict[Future[PreparedTile], TileKey],
    qualifying: dict[Future[TileResult], TileKey],
    prepared: dict[TileKey, PreparedTile],
    proxy_pool: ThreadPoolExecutor,
    candidate_pool: ThreadPoolExecutor,
) -> None:
    """Cancel queued work, join active workers, then dispose unused prepared images."""
    token.set()
    pending: tuple[Future[PreparedTile] | Future[TileResult], ...] = (*preparing, *qualifying)
    for future in pending:
        future.cancel()
    # Active tasks check cancellation between blocks of128 peaks.
    proxy_pool.shutdown(wait=True, cancel_futures=True)
    candidate_pool.shutdown(wait=True, cancel_futures=True)
    for image, _, _ in prepared.values():
        if image is not None:
            image.clear_cache()
    for prepared_future in preparing:
        if not prepared_future.cancelled() and prepared_future.exception() is None:
            image, _, _ = prepared_future.result()
            if image is not None:
                image.clear_cache()


def _compiled_tiles(
    width: int,
    height: int,
    read_tile: Callable[[Bounds], FloatArray],
    threshold: float,
    neighbor_radius: int,
    token: Event,
    coarse: bool,
    excluded: Sequence[CandidateRecord],
    sky_mask: SkyMask | None,
    source_scale: int,
) -> Generator[tuple[int, int, TileResult], None, None]:
    """Schedule bounded tile work and yield completed tiles in row-major join order."""
    visited: BoolArray = anonymous_array((height, width), np.bool_)
    columns, row_count = (width + TILE - 1) // TILE, (height + TILE - 1) // TILE
    keys: list[tuple[int, int]] = [
        (row_index, column_index)
        for row_index in range(row_count)
        for column_index in range(columns)
    ]
    results: dict[TileKey, TileResult] = {}
    completed: set[TileKey] = set()
    preparing: dict[Future[PreparedTile], TileKey] = {}
    prepared: dict[TileKey, PreparedTile] = {}
    qualifying: dict[Future[TileResult], TileKey] = {}
    next_join: int = 0
    next_prepare: int = 0
    prepare: partial[tuple[Image | None, list[tuple[int, int]], DoubleArray]] = partial(
        _prepare_tile,
        width=width,
        height=height,
        read_tile=read_tile,
        threshold=threshold,
        coarse=coarse,
        token=token,
        sky_mask=sky_mask,
        source_scale=source_scale,
    )
    qualify_tile: partial[tuple[list[CandidateRecord], int, int, DoubleArray]] = partial(
        _qualify_tile,
        threshold=threshold,
        neighbor_radius=neighbor_radius,
        coarse=coarse,
        excluded=excluded,
        visited=visited,
        token=token,
    )
    with (
        ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="star-proxy") as proxy_pool,
        ThreadPoolExecutor(
            max_workers=WORKERS, thread_name_prefix="star-candidate"
        ) as candidate_pool,
    ):
        try:
            while len(completed) < len(keys):
                check_cancel(token)
                while next_prepare < len(keys) and len(preparing) + len(prepared) + len(
                    qualifying
                ) < max(2, IN_FLIGHT):
                    key: tuple[int, int] = keys[next_prepare]
                    preparing[proxy_pool.submit(prepare, *key)] = key
                    next_prepare += 1
                for key in list(prepared):
                    if _tile_ready(*key, columns, completed):
                        qualifying[candidate_pool.submit(qualify_tile, prepared.pop(key))] = key
                _wait_tiles(preparing, qualifying, prepared, results, completed)
                while divmod(next_join, columns) in results:
                    value: tuple[list[CandidateRecord], int, int, DoubleArray] = results.pop(
                        divmod(next_join, columns)
                    )
                    yield next_join + 1, len(keys), value
                    next_join += 1
        except BaseException:
            _cancel_tiles(token, preparing, qualifying, prepared, proxy_pool, candidate_pool)
            raise


def scan_compiled(
    width: int,
    height: int,
    read_tile: Callable[[Bounds], FloatArray],
    threshold: float,
    neighbor_radius: int,
    cancel: Event | None,
    coarse: bool,
    excluded: Sequence[CandidateRecord],
    sky_mask: SkyMask | None,
    source_scale: int,
) -> tuple[list[CandidateRecord], Diagnostics]:
    """Join scheduled compiled tile catalogs deterministically while accumulating phase diagnostics."""
    from .join_fast import join_catalogs_fast

    token: Event = cancel if cancel is not None else Event()
    rows: list[CandidateRecord] = []
    peaks: int = 0
    seeds: int = 0
    catalog_maximum: float = 0.0
    phases: DoubleArray = np.zeros(3)
    join_seconds: float = 0.0
    tiles: Generator[
        tuple[int, int, tuple[list[CandidateRecord], int, int, DoubleArray]], None, None
    ] = _compiled_tiles(
        width,
        height,
        read_tile,
        threshold,
        neighbor_radius,
        token,
        coarse,
        excluded,
        sky_mask,
        source_scale,
    )
    # Closing also propagates join failures into the scheduler's cancellation cleanup.
    with closing(tiles):
        for next_join, tile_count, value in tiles:
            local, count_peaks, count_seeds, timing = value
            phases += timing
            joining: float = time.monotonic()
            catalog_maximum = join_catalogs_fast(rows, local, catalog_maximum)
            join_seconds += time.monotonic() - joining
            peaks += count_peaks
            seeds += count_seeds
            logger.info(
                "Compiled detector %s: %d/%d tiles, %d sources",
                "coarse" if coarse else "full",
                next_join,
                tile_count,
                len(rows),
            )
    return rows, dict(
        peaks=peaks,
        candidate_seeds=seeds,
        prepare_cpu_seconds=phases[0],
        seed_cpu_seconds=phases[1],
        qualify_cpu_seconds=phases[2],
        join_seconds=join_seconds,
    )


def _scan_coarse(
    codes: ByteArray,
    denominator: float,
    threshold: float,
    neighbor_radius: int,
    cancel: Event | None,
    sky_mask: SkyMask | None,
    source_shape: tuple[int, int],
) -> tuple[list[CandidateRecord], Diagnostics]:
    """Scan decoded coarse tiles and restore full-image source coordinates/sizes."""

    def coarse_tile(bounds: Bounds, source: ByteArray = codes) -> FloatArray:
        """Decode a coarse RGB8 tile using the image-wide normalization denominator."""
        left, top, right, bottom = bounds
        return decode(source[top:bottom, left:right], denominator)

    coarse_rows, coarse_stats = scan_tiled(
        codes.shape[1],
        codes.shape[0],
        coarse_tile,
        threshold,
        neighbor_radius,
        cancel,
        coarse=True,
        sky_mask=sky_mask,
        source_scale=4,
    )
    # Coarse pixel i averages a source box starting near i*step, where step is
    # (source - box)/(coarse - 1): exactly 4 only when the size is a multiple of 4.
    (box_y, step_y), (box_x, step_x) = (
        coarse_step(source, coarse) for source, coarse in zip(source_shape, codes.shape[:2])
    )
    for row in coarse_rows:
        row["x"] = (row["x"] - 0.5) * step_x + box_x / 2
        row["y"] = (row["y"] - 0.5) * step_y + box_y / 2
        row["size"] *= (step_x + step_y) / 2
    return coarse_rows, coarse_stats


def _normalize_rows(rows: list[CandidateRecord], sky_mask: SkyMask | None) -> list[CandidateRecord]:
    """Apply final mask filtering and one global normalization before catalog sorting."""
    if sky_mask is not None:
        keep: BoolArray = sky_mask.keep(
            np.array([(row["x"] - 0.5, row["y"] - 0.5) for row in rows])
        )
        rows = [row for row, allowed in zip(rows, keep) if allowed]
    colors, brightness, _ = normalize_catalog(
        [
            (candidate_record["rgb"], candidate_record["size"], candidate_record["brightness"])
            for candidate_record in rows
        ]
    )
    for row, color, value in zip(rows, colors, brightness):
        row["normalized_rgb"] = color
        row["normalized_brightness"] = value
    rows.sort(key=lambda row: (row["y"], row["x"]))
    return rows


def analyze(
    frame: ImageFrame,
    threshold: float = 0.2,
    neighbor_radius: int = 2,
    cancel: Event | None = None,
    sky_mask: SkyMask | None = None,
) -> tuple[list[CandidateRecord], Diagnostics]:
    """Build globally normalized coarse/full proxies and return source rows plus timing/count diagnostics."""
    start: float = time.monotonic()
    check_cancel(cancel)
    height, width = frame.pixels.shape[:2]
    use_coarse: bool = width > 256 or height > 256
    codes: ByteArray = (
        coarse_codes(frame.pixels, cancel, sky_mask) if use_coarse else full_codes(frame.pixels)
    )
    scale: int = 4 if use_coarse else 1
    allowed: None | BoolArray = (
        None
        if sky_mask is None
        else sky_mask.region((0, 0, codes.shape[1] * scale, codes.shape[0] * scale), sampling=scale)
    )
    denominator: float = maximum_luminance(codes, cancel, allowed)
    del allowed
    proxy_seconds: float = time.monotonic() - start
    logger.info(
        "Detector proxy %dx%d: global normalization %.8g; %.2fs",
        codes.shape[1],
        codes.shape[0],
        denominator,
        time.monotonic() - start,
    )
    coarse_rows: list[CandidateRecord] = []
    coarse_started: float = time.monotonic()
    coarse_stats: Diagnostics = dict(peaks=0, candidate_seeds=0)
    if use_coarse:
        coarse_rows, coarse_stats = _scan_coarse(
            codes, denominator, threshold, neighbor_radius, cancel, sky_mask, (height, width)
        )
    del codes
    coarse_seconds: float = time.monotonic() - coarse_started

    def full_tile(bounds: Bounds) -> FloatArray:
        """Read a decoded full-resolution RGB tile through the exact selected proxy path."""
        left, top, right, bottom = bounds
        pixels: FloatArray = frame.pixels[top:bottom, left:right]
        return (
            full_proxy(pixels, denominator) if COMPILED else decode(full_codes(pixels), denominator)
        )

    full_started: float = time.monotonic()
    full_rows, full_stats = scan_tiled(
        width,
        height,
        full_tile,
        threshold,
        neighbor_radius,
        cancel,
        excluded=coarse_rows,
        sky_mask=sky_mask,
    )
    rows: list[CandidateRecord] = coarse_rows + full_rows
    full_seconds: float = time.monotonic() - full_started
    rows = _normalize_rows(rows, sky_mask)
    return rows, dict(
        seconds=time.monotonic() - start,
        normalization_luminance=denominator,
        proxy_seconds=proxy_seconds,
        coarse_wall_seconds=coarse_seconds,
        full_wall_seconds=full_seconds,
        sky_mask=sky_mask is not None,
        coarse_count=len(coarse_rows),
        count=len(rows),
        coarse=coarse_stats,
        full=full_stats,
        method="Recovered candidate analysis with approximate local-maximum/neighbor seed selection; global normalization.",
    )
