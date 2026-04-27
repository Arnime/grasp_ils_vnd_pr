"""Tests covering ``GIVPConfig`` validation and direction logic."""

from __future__ import annotations

import pytest

from givp import GIVPConfig, InvalidConfigError


def test_default_config_is_valid():
    cfg = GIVPConfig()
    assert cfg.minimize is True
    assert cfg.direction == "minimize"


def test_minimize_false_sets_direction_maximize():
    cfg = GIVPConfig(minimize=False)
    assert cfg.direction == "maximize"


def test_direction_maximize_sets_minimize_false():
    cfg = GIVPConfig(direction="maximize")
    assert cfg.minimize is False


def test_invalid_direction_raises_invalid_config():
    with pytest.raises(InvalidConfigError):
        GIVPConfig(direction="bogus")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_iterations", 0),
        ("vnd_iterations", 0),
        ("ils_iterations", 0),
        ("elite_size", 0),
        ("path_relink_frequency", 0),
        ("num_candidates_per_step", 0),
        ("cache_size", 0),
        ("early_stop_threshold", 0),
        ("n_workers", 0),
    ],
)
def test_positive_int_fields_reject_zero(field, value):
    with pytest.raises(InvalidConfigError):
        GIVPConfig(**{field: value})


def test_perturbation_strength_negative_rejected():
    with pytest.raises(InvalidConfigError):
        GIVPConfig(perturbation_strength=-1)


@pytest.mark.parametrize("alpha", [-0.1, 1.1, 2.0])
def test_alpha_out_of_range_rejected(alpha):
    with pytest.raises(InvalidConfigError):
        GIVPConfig(alpha=alpha)


def test_alpha_min_greater_than_alpha_max_rejected():
    with pytest.raises(InvalidConfigError):
        GIVPConfig(alpha_min=0.5, alpha_max=0.1)


@pytest.mark.parametrize("field", ["alpha_min", "alpha_max"])
def test_alpha_bounds_out_of_range_rejected(field):
    with pytest.raises(InvalidConfigError):
        GIVPConfig(**{field: 1.5})  # type: ignore[arg-type]


def test_time_limit_negative_rejected():
    with pytest.raises(InvalidConfigError):
        GIVPConfig(time_limit=-1.0)


def test_integer_split_negative_rejected():
    with pytest.raises(InvalidConfigError):
        GIVPConfig(integer_split=-1)


def test_integer_split_none_allowed():
    cfg = GIVPConfig(integer_split=None)
    assert cfg.integer_split is None


def test_as_core_config_copies_fields():
    cfg = GIVPConfig(max_iterations=7, alpha=0.3)
    core_cfg = cfg.as_core_config()
    assert core_cfg.max_iterations == 7
    assert core_cfg.alpha == pytest.approx(0.3)
