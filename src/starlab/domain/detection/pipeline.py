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
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from threading import Event

import numpy as np
from scipy import ndimage

from ..imaging import check_cancel
from .candidate import Image, candidate, covered, merge_candidate
from .catalog import join_catalogs
from .numeric import normalize_catalog
from .peaks import peak_positions
from .proxy import coarse_codes, decode, full_codes, full_proxy, maximum_luminance

TILE = 1024
HALO = 272  # Covers native 266-pixel rays and the largest disk/ring support.
logger = logging.getLogger(__name__)
COMPILED = True


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
            local_rows = []
            maximum_size = 0.0
            check_cancel(cancel)
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
            if allowed is not None and not allowed.any():
                continue
            image = Image(read_tile(bounds), origin=bounds[:2], allowed=allowed)
            try:
                for px, py in peak_positions_reference(image, threshold, core, coarse):
                    peaks += 1
                    check_cancel(cancel)
                    if covered(px + 0.5, py + 0.5, excluded):
                        continue
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
                        seeds += 1
                        result = candidate(image, x + 0.5, y + 0.5, coarse, excluded)
                        attempts.append((y, x, result))
                        if result is not None:
                            accepted = True
                            break
                    if accepted:
                        for y in range(
                            max(0, py - neighbor_radius), min(height, py + neighbor_radius + 1)
                        ):
                            for x in range(
                                max(0, px - neighbor_radius), min(width, px + neighbor_radius + 1)
                            ):
                                if (x, y) in visited or covered(x + 0.5, y + 0.5, excluded):
                                    continue
                                if image.rgb[y - image.oy, x - image.ox].max() < threshold:
                                    continue
                                visited.add((x, y))
                                seeds += 1
                                result = candidate(image, x + 0.5, y + 0.5, coarse, excluded)
                                attempts.append((y, x, result))
                    # Native local merging scans seed rows left-to-right.
                    for _, _, result in sorted(attempts, key=lambda row: row[:2]):
                        if result is not None:
                            maximum_size = merge_candidate(local_rows, result, maximum_size)
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
    from .compiled import scan_tile
    from .join_fast import join_catalogs_fast

    token = cancel if cancel is not None else Event()
    visited = np.zeros((height, width), np.bool_)
    columns, row_count = (width + TILE - 1) // TILE, (height + TILE - 1) // TILE
    keys = [(r, c) for r in range(row_count) for c in range(columns)]
    results, completed, preparing, prepared, qualifying = {}, set(), {}, {}, {}
    rows = []
    peaks = seeds = next_join = next_prepare = 0
    catalog_maximum = 0.0
    phases = np.zeros(3)
    join_seconds = 0.0

    def prepare(row, column):
        started = time.monotonic()
        check_cancel(token)
        left, top = column * TILE, row * TILE
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

    def qualify_tile(value):
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

    def ready(row, column):
        # Preparation has no shared writes. Only qualification observes visited.
        # Preserve all adjacent writes; merge catalogues later in row-major order.
        return (column == 0 or (row, column - 1) in completed) and (
            row == 0 or (row - 1, min(column + 1, columns - 1)) in completed
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
                    if ready(*key):
                        qualifying[candidate_pool.submit(qualify_tile, prepared.pop(key))] = key
                finished, _ = wait((*preparing, *qualifying), return_when=FIRST_COMPLETED)
                for future in finished:
                    if future in preparing:
                        prepared[preparing.pop(future)] = future.result()
                    else:
                        key = qualifying.pop(future)
                        results[key] = future.result()
                        completed.add(key)
                while divmod(next_join, columns) in results:
                    local, count_peaks, count_seeds, timing = results.pop(
                        divmod(next_join, columns)
                    )
                    phases += timing
                    joining = time.monotonic()
                    catalog_maximum = join_catalogs_fast(rows, local, catalog_maximum)
                    join_seconds += time.monotonic() - joining
                    peaks += count_peaks
                    seeds += count_seeds
                    next_join += 1
                    logger.info(
                        "Compiled detector %s: %d/%d tiles, %d sources",
                        "coarse" if coarse else "full",
                        next_join,
                        len(keys),
                        len(rows),
                    )
        except BaseException:
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
            raise
    return rows, dict(
        peaks=peaks,
        candidate_seeds=seeds,
        prepare_cpu_seconds=phases[0],
        seed_cpu_seconds=phases[1],
        qualify_cpu_seconds=phases[2],
        join_seconds=join_seconds,
    )


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
        del coarse_tile
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
