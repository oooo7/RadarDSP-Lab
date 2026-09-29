"""Sanity test checking that all package modules can be imported without errors."""

import importlib
import pytest

MODULES_TO_TEST = [
    "src",
    "src.dsp",
    "src.dsp.signals",
    "src.dsp.sampling",
    "src.dsp.transforms",
    "src.dsp.filtering",
    "src.radar",
    "src.radar.chirp",
    "src.radar.target",
    "src.radar.propagation",
    "src.radar.processing",
    "src.radar.cfar",
    "src.validation",
    "src.validation.metrics",
    "src.validation.monte_carlo",
    "src.utils",
    "src.utils.config",
    "src.utils.logging",
    "app",
    "app.views.dsp_lab",
    "app.views.radar_lab",
]


@pytest.mark.parametrize("module_name", MODULES_TO_TEST)
def test_module_import(module_name: str) -> None:
    """Verify that each module in the architecture imports cleanly."""
    module = importlib.import_module(module_name)
    assert module is not None
