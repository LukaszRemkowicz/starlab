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
from collections.abc import Callable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import closing
from functools import partial
from threading import Event

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from ..imaging import check_cancel
from ..sky_mask import SkyMask
from .candidate import Image, candidate, covered, merge_candidate
from .catalog import join_catalogs
from .numeric import normalize_catalog
from .peaks import peak_positions
from .proxy import coarse_codes, decode, full_codes, full_proxy, maximum_luminance

TILE = 1024
HALO = 272  # Covers native 266-pixel rays and the largest disk/ring support.
logger = logging.getLogger(__name__)
COMPILED = True


def _tile_region(
    width: int,
    height: int,
    left: int,
    top: int,
    sky_mask: SkyMask | None,
    source_scale: int,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], NDArray[np.bool_] | None]:
    """Return tile core, clipped halo bounds, and allowed pixels at source scale."""
    core = (left, top, min(width, left + TILE), min(height, top + TILE))
    bounds = (
        max(0, left - HALO),
        max(0, top - HALO),
        min(width, core[2] + HALO),
        min(height, core[3] + HALO),
    )
    allowed = (
        None
        if sky_mask is None
        else sky_mask.region(tuple(v * source_scale for v in bounds), sampling=source_scale)
    )
    return core, bounds, allowed


def peak_positions_reference(image, threshold, core, coarse=False):
    intensity = image.rgb.max(axis=2)
    allowed = image.allowed
    maxima = ndimage.maximum_filter(
        image.lum if allowed is None else np.where(allowed, image.lum, -np.inf),
        size=3,
        mode="nearest",
    )
    floor = ndimage.minimum_filter(
        image.lum if allowed is None else np.where(allowed, image.lum, np.inf),
        size=129,
        mode="nearest",
    )
    support = ndimage.uniform_filter(
        ((intensity > threshold * 0.05) & (True if allowed is None else allowed)).astype(
            np.float32
        ),
        size=3,
        mode="constant",
    )
    selected = (
        (image.lum == maxima)
        & (intensity >= threshold)
        & (image.lum - floor >= 0.13)
        & (support >= 2.5 / 9)
    )
    if allowed is not None:
        selected &= allowed
    if coarse:
        # This pass keeps only size>4. Native98440 requires center maximum>=.75
        # and normalized inner-disk score>=.67. The3x3 mean is a deliberately
        # approximate early screen for that broad bright support, not a rank.
        selected &= (image.lum >= 0.75) & (
            ndimage.uniform_filter(image.lum, size=3, mode="nearest") >= 0.6
        )
        contrast_allowed = None
    else:
        # Approximate early screen inspired by the recovered mean-std shape
        # score. A local peak on a textured bright background can satisfy the
        # distant minimum test while having no compact source contrast.
        values = image.lum if allowed is None else np.where(allowed, image.lum, 0)
        mean = ndimage.uniform_filter(values, size=7, mode="nearest")
        square = ndimage.uniform_filter(values**2, size=7, mode="nearest")
        if allowed is not None:
            count = ndimage.uniform_filter(allowed.astype(np.float32), size=7, mode="nearest")
            mean = np.divide(mean, count, out=np.zeros_like(mean), where=count > 0)
            square = np.divide(square, count, out=np.zeros_like(square), where=count > 0)
        std = np.sqrt(np.maximum(square - mean**2, 0))
        contrast_allowed = (image.lum >= 0.75) | (image.lum - mean + std >= 0.08)
    labels, _ = ndimage.label(selected)
    for region in ndimage.find_objects(labels):
        if region is None:
            continue
        y = (region[0].start + region[0].stop - 1) // 2
        x = (region[1].start + region[1].stop - 1) // 2
        if contrast_allowed is not None and not contrast_allowed[y, x]:
            continue
        x += image.ox
        y += image.oy
        if core[0] <= x < core[2] and core[1] <= y < core[3]:
            yield x, y


def _reference_neighbors(
    image: Image,
    px: int,
    py: int,
    width: int,
    height: int,
    threshold: float,
    neighbor_radius: int,
    coarse: bool,
    excluded: tuple | list,
    visited: set[tuple[int, int]],
) -> list[tuple[int, int, dict | None]]:
    """Evaluate unvisited neighborhood seeds in native row order."""
    attempts = []
    for y in range(max(0, py - neighbor_radius), min(height, py + neighbor_radius + 1)):
        for x in range(max(0, px - neighbor_radius), min(width, px + neighbor_radius + 1)):
            if (x, y) in visited or covered(x + 0.5, y + 0.5, excluded):
                continue
            if image.rgb[y - image.oy, x - image.ox].max() < threshold:
                continue
            visited.add((x, y))
            result = candidate(image, x + 0.5, y + 0.5, coarse, excluded)
            attempts.append((y, x, result))
    return attempts


def _reference_attempts(
    image: Image,
    px: int,
    py: int,
    width: int,
    height: int,
    threshold: float,
    neighbor_radius: int,
    coarse: bool,
    excluded: tuple | list,
    visited: set[tuple[int, int]],
) -> list[tuple[int, int, dict | None]]:
    """Try the preferred basin seeds, then expand only after an accepted candidate."""
    accepted = False
    # A maximum may itself fail the original shape predicate;
    # four adjacent seeds retain the nearby qualifying basin.
    attempts = []
    for dx, dy in (
        (0, 0),
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (-1, 1),
        (-1, -1),
        (1, -1),
    ):
        x, y = px + dx, py + dy
        if not (0 <= x < width and 0 <= y < height):
            continue
        if (x, y) in visited:
            continue
        if image.rgb[y - image.oy, x - image.ox].max() < threshold:
            continue
        visited.add((x, y))
        result = candidate(image, x + 0.5, y + 0.5, coarse, excluded)
        attempts.append((y, x, result))
        if result is not None:
            accepted = True
            break
    if accepted:
        attempts.extend(
            _reference_neighbors(
                image, px, py, width, height, threshold, neighbor_radius, coarse, excluded, visited
            )
        )
    return attempts


def _scan_reference_tile(
    image: Image,
    core: tuple[int, int, int, int],
    width: int,
    height: int,
    threshold: float,
    neighbor_radius: int,
    coarse: bool,
    excluded: tuple | list,
    visited: set[tuple[int, int]],
    cancel: Event | None,
) -> tuple[list[dict], int, int]:
    """Qualify a reference tile and merge accepted attempts in native seed order."""
    local_rows = []
    maximum_size = 0.0
    peaks = seeds = 0
    for px, py in peak_positions_reference(image, threshold, core, coarse):
        peaks += 1
        check_cancel(cancel)
        if covered(px + 0.5, py + 0.5, excluded):
            continue
        attempts = _reference_attempts(
            image, px, py, width, height, threshold, neighbor_radius, coarse, excluded, visited
        )
        seeds += len(attempts)
        # Native local merging scans seed rows left-to-right.
        for _, _, result in sorted(attempts, key=lambda row: row[:2]):
            if result is not None:
                maximum_size = merge_candidate(local_rows, result, maximum_size)
    return local_rows, peaks, seeds


def scan_tiled(
    width,
    height,
    read_tile,
    threshold,
    neighbor_radius,
    cancel,
    coarse=False,
    excluded=(),
    sky_mask=None,
    source_scale=1,
):
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
    rows = []
    visited = set()
    peaks = seeds = 0
    for top in range(0, height, TILE):
        for left in range(0, width, TILE):
            tile_started = time.monotonic()
            check_cancel(cancel)
            core, bounds, allowed = _tile_region(width, height, left, top, sky_mask, source_scale)
            if allowed is not None and not allowed.any():
                continue
            image = Image(read_tile(bounds), origin=bounds[:2], allowed=allowed)
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


def _prepare_tile(
    row: int,
    column: int,
    *,
    width: int,
    height: int,
    read_tile: Callable[[tuple[int, int, int, int]], NDArray[np.float32]],
    threshold: float,
    coarse: bool,
    token: Event,
    sky_mask: SkyMask | None,
    source_scale: int,
) -> tuple[Image | None, list[tuple[int, int]], NDArray[np.float64]]:
    """Read one halo tile and locate peaks without writing shared visitation state."""
    started = time.monotonic()
    check_cancel(token)
    left, top = column * TILE, row * TILE
    core, bounds, allowed = _tile_region(width, height, left, top, sky_mask, source_scale)
    if allowed is not None and not allowed.any():
        return None, [], np.zeros(2)
    image = Image(read_tile(bounds), origin=bounds[:2], allowed=allowed)
    converted = time.monotonic()
    try:
        positions = list(peak_positions(image, threshold, core, coarse))
        return image, positions, np.array([converted - started, time.monotonic() - converted])
    except BaseException:
        image.clear_cache()
        raise


def _qualify_tile(
    value: tuple[Image | None, list[tuple[int, int]], NDArray[np.float64]],
    *,
    threshold: float,
    neighbor_radius: int,
    coarse: bool,
    excluded: tuple | list,
    visited: NDArray[np.bool_],
    token: Event,
) -> tuple[list[dict], int, int, NDArray[np.float64]]:
    """Qualify a prepared tile and release its caches even on cancellation."""
    from .compiled import scan_tile

    image, positions, timing = value
    if image is None:
        return [], 0, 0, np.zeros(3)
    started = time.monotonic()
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


def _tile_ready(
    row: int,
    column: int,
    columns: int,
    completed: set[tuple[int, int]],
) -> bool:
    """Preserve adjacent visitation writes before a tile starts qualification."""
    return (column == 0 or (row, column - 1) in completed) and (
        row == 0 or (row - 1, min(column + 1, columns - 1)) in completed
    )


def _wait_tiles(
    preparing: dict[Future, tuple[int, int]],
    qualifying: dict[Future, tuple[int, int]],
    prepared: dict[tuple[int, int], tuple],
    results: dict[tuple[int, int], tuple],
    completed: set[tuple[int, int]],
) -> None:
    """Collect completed preparation/qualification work without changing join order."""
    finished, _ = wait((*preparing, *qualifying), return_when=FIRST_COMPLETED)
    for future in finished:
        if future in preparing:
            prepared[preparing.pop(future)] = future.result()
        else:
            key = qualifying.pop(future)
            results[key] = future.result()
            completed.add(key)


def _cancel_tiles(
    token: Event,
    preparing: dict[Future, tuple[int, int]],
    qualifying: dict[Future, tuple[int, int]],
    prepared: dict[tuple[int, int], tuple],
    proxy_pool: ThreadPoolExecutor,
    candidate_pool: ThreadPoolExecutor,
) -> None:
    """Cancel queued work, join active workers, then dispose unused prepared images."""
    token.set()
    for future in (*preparing, *qualifying):
        future.cancel()
    # Active tasks check cancellation between blocks of128 peaks.
    proxy_pool.shutdown(wait=True, cancel_futures=True)
    candidate_pool.shutdown(wait=True, cancel_futures=True)
    for image, _, _ in prepared.values():
        if image is not None:
            image.clear_cache()
    for future in preparing:
        if not future.cancelled() and future.exception() is None:
            image, _, _ = future.result()
            if image is not None:
                image.clear_cache()


def _compiled_tiles(
    width: int,
    height: int,
    read_tile: Callable[[tuple[int, int, int, int]], NDArray[np.float32]],
    threshold: float,
    neighbor_radius: int,
    token: Event,
    coarse: bool,
    excluded: tuple | list,
    sky_mask: SkyMask | None,
    source_scale: int,
) -> Iterator[tuple[int, int, tuple[list[dict], int, int, NDArray[np.float64]]]]:
    """Schedule bounded tile work and yield completed tiles in row-major join order."""
    visited = np.zeros((height, width), np.bool_)
    columns, row_count = (width + TILE - 1) // TILE, (height + TILE - 1) // TILE
    keys = [(r, c) for r in range(row_count) for c in range(columns)]
    results, completed, preparing, prepared, qualifying = {}, set(), {}, {}, {}
    next_join = next_prepare = 0
    prepare = partial(
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
    qualify_tile = partial(
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
                ) < max(2, WORKERS * 2):
                    key = keys[next_prepare]
                    preparing[proxy_pool.submit(prepare, *key)] = key
                    next_prepare += 1
                for key in list(prepared):
                    if _tile_ready(*key, columns, completed):
                        qualifying[candidate_pool.submit(qualify_tile, prepared.pop(key))] = key
                _wait_tiles(preparing, qualifying, prepared, results, completed)
                while divmod(next_join, columns) in results:
                    value = results.pop(divmod(next_join, columns))
                    yield next_join + 1, len(keys), value
                    next_join += 1
        except BaseException:
            _cancel_tiles(token, preparing, qualifying, prepared, proxy_pool, candidate_pool)
            raise


def scan_compiled(
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
):
    from .join_fast import join_catalogs_fast

    token = cancel if cancel is not None else Event()
    rows = []
    peaks = seeds = 0
    catalog_maximum = 0.0
    phases = np.zeros(3)
    join_seconds = 0.0
    tiles = _compiled_tiles(
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
            joining = time.monotonic()
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
    codes: NDArray[np.uint8],
    denominator: float,
    threshold: float,
    neighbor_radius: int,
    cancel: Event | None,
    sky_mask: SkyMask | None,
) -> tuple[list[dict], dict]:
    """Scan decoded coarse tiles and restore full-image source coordinates/sizes."""

    def coarse_tile(bounds, source=codes):
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
    for row in coarse_rows:
        for key in ("x", "y", "size"):
            row[key] *= 4
    return coarse_rows, coarse_stats


def _normalize_rows(rows: list[dict], sky_mask: SkyMask | None) -> list[dict]:
    """Apply final mask filtering and one global normalization before catalog sorting."""
    if sky_mask is not None:
        keep = sky_mask.keep(np.array([(row["x"] - 0.5, row["y"] - 0.5) for row in rows]))
        rows = [row for row, allowed in zip(rows, keep) if allowed]
    colors, brightness, _ = normalize_catalog(
        [(r["rgb"], r["size"], r["brightness"]) for r in rows]
    )
    for row, color, value in zip(rows, colors, brightness):
        row["normalized_rgb"] = color
        row["normalized_brightness"] = value
    rows.sort(key=lambda row: (row["y"], row["x"]))
    return rows


def analyze(frame, threshold=0.2, neighbor_radius=2, cancel=None, sky_mask=None):
    start = time.monotonic()
    check_cancel(cancel)
    height, width = frame.pixels.shape[:2]
    use_coarse = width > 256 or height > 256
    codes = coarse_codes(frame.pixels, cancel, sky_mask) if use_coarse else full_codes(frame.pixels)
    scale = 4 if use_coarse else 1
    allowed = (
        None
        if sky_mask is None
        else sky_mask.region((0, 0, codes.shape[1] * scale, codes.shape[0] * scale), sampling=scale)
    )
    denominator = maximum_luminance(codes, cancel, allowed)
    del allowed
    proxy_seconds = time.monotonic() - start
    logger.info(
        "Detector proxy %dx%d: global normalization %.8g; %.2fs",
        codes.shape[1],
        codes.shape[0],
        denominator,
        time.monotonic() - start,
    )
    coarse_rows = []
    coarse_started = time.monotonic()
    coarse_stats = dict(peaks=0, candidate_seeds=0)
    if use_coarse:
        coarse_rows, coarse_stats = _scan_coarse(
            codes, denominator, threshold, neighbor_radius, cancel, sky_mask
        )
    del codes
    coarse_seconds = time.monotonic() - coarse_started

    def full_tile(bounds):
        left, top, right, bottom = bounds
        pixels = frame.pixels[top:bottom, left:right]
        return (
            full_proxy(pixels, denominator) if COMPILED else decode(full_codes(pixels), denominator)
        )

    full_started = time.monotonic()
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
    rows = coarse_rows + full_rows
    full_seconds = time.monotonic() - full_started
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
