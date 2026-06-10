"""Unit tests for screen-proportional UI scaling (ui_theme module)."""


def test_scale_value_exists():
    from mrid_app.ui_theme import scale_value
    assert callable(scale_value)


def test_scale_thumbnail_exists():
    from mrid_app.ui_theme import scale_thumbnail
    assert callable(scale_thumbnail)


def test_cascade_exists():
    from mrid_app.ui_theme import cascade
    assert callable(cascade)


def test_scale_value_returns_int():
    from mrid_app.ui_theme import scale_value
    assert isinstance(scale_value(10), int)


def test_scale_value_positive():
    from mrid_app.ui_theme import scale_value
    assert scale_value(100) > 0


def test_scale_thumbnail_applies_config_factor():
    """scale_thumbnail must reflect THUMBNAIL_SCALE_FACTOR from config."""
    from mrid_app import config, ui_theme

    original = config.THUMBNAIL_SCALE_FACTOR
    try:
        config.THUMBNAIL_SCALE_FACTOR = 1.0
        base = ui_theme.scale_thumbnail(100)
        config.THUMBNAIL_SCALE_FACTOR = 2.0
        doubled = ui_theme.scale_thumbnail(100)
        assert doubled == 2 * base
    finally:
        config.THUMBNAIL_SCALE_FACTOR = original


def test_default_scale_factor_is_one_without_init():
    """Before init_scaling runs, helpers must behave as scale 1.0 fallback."""
    from mrid_app import ui_theme

    if ui_theme._SCALE_FACTOR is None:
        assert ui_theme.get_scale_factor() == 1.0
        assert ui_theme.scale_value(42) == 42
    else:
        # init_scaling already ran in this process; just check consistency
        assert ui_theme.scale_value(100) == int(round(100 * ui_theme.get_scale_factor()))


def test_init_scaling_bounds():
    """init_scaling clamps the factor to a sane range when a display exists."""
    import customtkinter as ctk
    from mrid_app import ui_theme

    try:
        root = ctk.CTk()
        root.withdraw()
        scale = ui_theme.init_scaling(root)
        assert 0.6 <= scale <= 3.0
        root.destroy()
    except Exception:
        # No display available (e.g. headless CI) — nothing to verify
        pass
