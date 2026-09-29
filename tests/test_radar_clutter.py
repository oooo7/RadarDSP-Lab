"""Unit tests for Radar Statistical Clutter Module (src/radar/clutter.py)."""

import numpy as np
import pytest

from src.radar.clutter import (
    ClutterResult,
    add_clutter,
    generate_clutter,
)


def test_clutter_generation_shape_and_power() -> None:
    """Test 1: Verify clutter generator output shape, complex data type, and power scaling."""
    shape = (64, 500)
    target_p_clutter = 2.0

    c_res = generate_clutter(shape=shape, clutter_power=target_p_clutter, seed=42)

    assert isinstance(c_res, ClutterResult)
    assert c_res.clutter_signal.shape == shape
    assert np.iscomplexobj(c_res.clutter_signal)
    # Measured power should match target clutter power within finite-sample statistics
    assert c_res.clutter_power == pytest.approx(target_p_clutter, abs=0.2)


def test_clutter_deterministic_seed() -> None:
    """Test 2: Verify deterministic clutter generation with random seed."""
    c1 = generate_clutter(shape=(100, 100), clutter_power=1.0, seed=77)
    c2 = generate_clutter(shape=(100, 100), clutter_power=1.0, seed=77)
    c3 = generate_clutter(shape=(100, 100), clutter_power=1.0, seed=88)

    np.testing.assert_array_equal(c1.clutter_signal, c2.clutter_signal)
    assert not np.array_equal(c1.clutter_signal, c3.clutter_signal)


def test_add_clutter_to_signal() -> None:
    """Test 3: Verify add_clutter function adds complex clutter payload."""
    signal = np.ones((32, 200), dtype=np.complex128)
    cluttered_sig, c_res = add_clutter(signal, clutter_power=0.5, seed=123)

    assert cluttered_sig.shape == (32, 200)
    assert np.iscomplexobj(cluttered_sig)
    assert c_res.target_clutter_power == 0.5


def test_invalid_clutter_parameters() -> None:
    """Test 4: Verify ValueError exceptions on invalid clutter inputs."""
    with pytest.raises(ValueError, match="strictly positive"):
        generate_clutter(shape=(10, 10), clutter_power=-1.0)

    with pytest.raises(ValueError, match="Shape dimensions"):
        generate_clutter(shape=(0, 10))
