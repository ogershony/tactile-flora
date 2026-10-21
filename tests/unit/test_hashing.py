from pathlib import Path

import numpy as np

from tactile_flora.verify import hashing
from tests.fixtures.synthetic import make_pad_episode
from tests.fixtures.synthetic_video import SyntheticFrameReader, smooth_image


def test_identical_frames_distance_zero() -> None:
    img = smooth_image("a")
    h = hashing.phash64(img)
    assert len(h) == 16 and hashing.hamming(h, hashing.phash64(img.copy())) == 0


def test_shifted_noisy_small_distance() -> None:
    img = smooth_image("a")
    rng = np.random.default_rng(0)
    near = np.roll(img, 1, axis=1) + rng.normal(0, 2, img.shape)
    assert hashing.hamming(hashing.phash64(img), hashing.phash64(near)) <= 12
    assert hashing.hamming(hashing.phash64(img), hashing.phash64(near)) < hashing.hamming(
        hashing.phash64(img), hashing.phash64(smooth_image("zzz"))
    )


def test_random_noise_large_distance() -> None:
    rng = np.random.default_rng(1)
    a = rng.uniform(0, 255, (64, 64))
    b = rng.uniform(0, 255, (64, 64))
    assert hashing.hamming(hashing.phash64(a), hashing.phash64(b)) > 20
    assert (
        hashing.hamming(hashing.phash64(smooth_image("a")), hashing.phash64(smooth_image("b"))) > 20
    )


def test_32x32_input_and_rgb() -> None:
    img = smooth_image("a", 32)
    rgb = np.repeat(img[:, :, None], 3, axis=2)
    assert hashing.hamming(hashing.phash64(img), hashing.phash64(rgb)) <= 2


def test_index_roundtrip_and_overlap(tmp_path: Path) -> None:
    reader = SyntheticFrameReader()
    bench = [make_pad_episode(s).model_copy(update={"source": "bench"}) for s in (10, 11)]
    path = hashing.build_benchmark_index(bench, reader, tmp_path / "v" / "benchmark_hashes.parquet")
    index = hashing.load_index(path)
    assert [r.episode_id for r in index] == ["pad-10", "pad-11"]
    assert hashing.load_index(tmp_path / "missing.parquet") == []
    dup = make_pad_episode(5)
    other = SyntheticFrameReader(alias={"pad-5": "pad-11"})
    assert hashing.is_overlap(hashing.episode_hashes(dup, other), index)
    assert not hashing.is_overlap(hashing.episode_hashes(dup, reader), index)
    # all three must match the same row: mixing rows does not count
    h10 = hashing.episode_hashes(bench[0], reader)
    h11 = hashing.episode_hashes(bench[1], reader)
    assert not hashing.is_overlap([h10[0], h11[1], h11[2]], index)


def test_hash_cache(tmp_path: Path) -> None:
    reader = SyntheticFrameReader()
    ep = make_pad_episode(3)
    cache: dict[str, list[str]] = {}
    first = hashing.cached_hashes(ep, reader, cache)
    assert cache == {"pad-3": first}
    p = hashing.save_hash_cache(tmp_path / "s" / "hashes.parquet", cache)
    assert hashing.load_hash_cache(p) == cache
    # a cached episode is not decoded again
    assert hashing.cached_hashes(ep, SyntheticFrameReader(fail={"pad-3"}), cache) == first
    assert hashing.load_hash_cache(tmp_path / "none.parquet") == {}
