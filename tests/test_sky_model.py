import numpy as np

from starlab.domain.imaging import ImageFrame
from starlab.infrastructure.sky_mask import OnnxSkySegmenter, SkyMaskService


def test_packaged_model_matches_the_starligner_recorded_logits():
    net = OnnxSkySegmenter()
    ramp = np.add.outer(np.linspace(-1, 3, 256), np.linspace(0, 2, 256))[:, :, None]
    wave = np.sin(np.arange(256 * 256 * 3).reshape(256, 256, 3) * 0.37) * 0.3
    tile = (ramp * np.array([1, 0.8, 0.6]) + wave).astype(np.float32)
    logits = net.sky_logits(tile)
    assert net.identity == "b39141ebe7339f3aec897c7da058a935032540f80c1da04d7abf7fc1650d606d"
    np.testing.assert_allclose(
        logits[[0, 17, 100, 255, 128], [3, 200, 100, 255, 128]],
        [1.011243, 4.649025, 4.783614, 3.19662, 5.819427],
        atol=1e-4,
    )
    np.testing.assert_array_equal(logits, net.sky_logits(tile))


def test_batched_model_preserves_single_tile_logits():
    net = OnnxSkySegmenter()
    tiles = np.random.default_rng(9).uniform(-1, 8, (3, 256, 256, 3)).astype(np.float32)
    expected = np.stack([net.sky_logits(tile) for tile in tiles])
    np.testing.assert_allclose(net.sky_logits_batch(tiles), expected, atol=1e-4, rtol=1e-5)


def test_native_block_means_preserve_encoded_samples():
    from starlab.infrastructure.sky_mask import _block_mean

    rng = np.random.default_rng(7)
    for dtype in (np.uint8, np.uint16, ">u2", np.float32):
        for channels in (1, 3):
            for factor in (2, 3, 4, 5, 8, 16, 17):
                image = rng.integers(0, 65536, (factor * 3, factor * 7, channels)).astype(dtype)
                for view in (image, image[:, ::-1]):
                    expected = view.reshape(3, factor, 7, factor, channels).mean(
                        axis=(1, 3), dtype=np.float32
                    )
                    np.testing.assert_array_equal(_block_mean(view, factor), expected)


def test_mask_service_reads_encoded_tiff_and_reuses_disk_cache(tmp_path):
    import tifffile

    source = tmp_path / "sky.tif"
    data = np.zeros((16, 24, 3), np.uint16)
    data[:8] = 40000
    data[8:] = 12000
    tifffile.imwrite(source, data, photometric="rgb")

    class Network:
        tile = 8
        overlap = 2
        identity = "test-model"
        calls = 0

        def sky_logits(self, tile):
            self.calls += 1
            return np.ones(tile.shape[:2], np.float32) * 2

    net = Network()
    service = SkyMaskService(cache_directory=tmp_path / "cache", segmenter=net)
    # Different linear pixels prove the source TIFF is used for the model input.
    frame = ImageFrame(np.ones(data.shape, np.float32))
    mask = service(frame, source)
    assert net.calls > 0 and mask.classes.shape == (8, 12)

    class UnusedNetwork(Network):
        def sky_logits(self, tile):
            raise AssertionError("cached mask must not rerun the network")

    again = SkyMaskService(cache_directory=tmp_path / "cache", segmenter=UnusedNetwork())(
        frame, source
    )
    np.testing.assert_array_equal(again.classes, mask.classes)


def test_disabled_persistent_cache_recalculates_without_overwriting_results(tmp_path):
    import tifffile

    source = tmp_path / "fresh.tif"
    tifffile.imwrite(source, np.zeros((16, 16, 3), np.uint16), photometric="rgb")

    class Network:
        tile, overlap, identity = 8, 2, "same-model-for-cache-control-test"

        def sky_logits(self, tile):
            return np.full(tile.shape[:2], 2, np.float32)

    cache = tmp_path / "cache"
    frame = ImageFrame(np.zeros((16, 16, 3), np.float32))
    first = SkyMaskService(cache_directory=cache, segmenter=Network())(frame, source)
    assert (first.classes == 1).all()
    stored = {p.name: p.read_bytes() for p in cache.iterdir()}

    class ForegroundNetwork(Network):
        def sky_logits(self, tile):
            return np.full(tile.shape[:2], -2, np.float32)

    actual = SkyMaskService(
        cache_directory=cache, segmenter=ForegroundNetwork(), persistent_cache=False
    )(frame, source)
    assert (actual.classes == 0).all()
    assert {p.name: p.read_bytes() for p in cache.iterdir()} == stored
