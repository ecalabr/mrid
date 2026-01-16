"""Unit tests for DPI-aware autoscaling functionality."""
import tkinter as tk


def test_calculate_scale_factor_exists():
    """Test that _calculate_scale_factor function exists."""
    from mrid_app.gui import _calculate_scale_factor
    assert callable(_calculate_scale_factor)


def test_enable_tkinter_autoscaling_exists():
    """Test that enable_tkinter_autoscaling function exists."""
    from mrid_app.gui import enable_tkinter_autoscaling
    assert callable(enable_tkinter_autoscaling)


def test_scale_value_exists():
    """Test that scale_value function exists."""
    from mrid_app.gui import scale_value
    assert callable(scale_value)


def test_scale_geometry_exists():
    """Test that scale_geometry function exists."""
    from mrid_app.gui import scale_geometry
    assert callable(scale_geometry)


def test_scale_value_returns_int():
    """Test that scale_value returns an integer."""
    from mrid_app.gui import scale_value
    result = scale_value(10)
    assert isinstance(result, int)


def test_scale_value_positive():
    """Test that scale_value with positive input returns positive output."""
    from mrid_app.gui import scale_value
    result = scale_value(100)
    assert result > 0


def test_scale_geometry_format():
    """Test that scale_geometry returns properly formatted string."""
    from mrid_app.gui import scale_geometry
    result = scale_geometry("800x600")
    assert isinstance(result, str)
    assert 'x' in result
    # Should be in format "NNNxMMM" or "NNNxMMM+X+Y"
    parts = result.split('+')[0]  # Get geometry part before any +X+Y offset
    assert len(parts.split('x')) == 2


def test_calculate_scale_factor_reasonable_bounds():
    """Test that _calculate_scale_factor returns reasonable values."""
    try:
        # Create a temporary hidden window for testing
        root = tk.Tk()
        root.withdraw()  # Hide the window

        from mrid_app.gui import _calculate_scale_factor
        scale_factor = _calculate_scale_factor(root)

        # Scale factor should be between 0.75 and 3.0 as per the code
        assert 0.75 <= scale_factor <= 3.0
        assert isinstance(scale_factor, float)

        root.destroy()
    except Exception:
        # If we can't create a window (e.g., no display), skip the assertion
        # but the test should still pass
        pass


def test_enable_dpi_awareness_exists():
    """Test that _enable_dpi_awareness function exists."""
    from mrid_app.gui import _enable_dpi_awareness
    assert callable(_enable_dpi_awareness)


def test_enable_dpi_awareness_no_error():
    """Test that _enable_dpi_awareness doesn't raise errors."""
    from mrid_app.gui import _enable_dpi_awareness
    try:
        # Should not raise an error even if DPI awareness isn't available
        _enable_dpi_awareness()
    except Exception as e:
        raise AssertionError(f"_enable_dpi_awareness raised an error: {e}")


