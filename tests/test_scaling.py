"""Test script to verify DPI-aware autoscaling functionality."""
import tkinter as tk
from mrid_app.gui import enable_tkinter_autoscaling, scale_geometry, scale_value, _SCALE_FACTOR


def main():
    root = tk.Tk()
    root.title("DPI Scaling Test")

    # Enable autoscaling
    enable_tkinter_autoscaling(root)

    # Display scale factor information
    scale_info = f"Detected Scale Factor: {_SCALE_FACTOR:.2f}"

    # Get DPI info
    try:
        dpi = root.winfo_fpixels('1i')
        dpi_info = f"Screen DPI: {dpi:.1f}"
    except:
        dpi_info = "Screen DPI: Unknown"

    # Create test window with scaled geometry
    root.geometry(scale_geometry("600x400"))

    # Create frame with scaled padding
    frame = tk.Frame(root, padx=scale_value(20), pady=scale_value(20))
    frame.pack(fill="both", expand=True)

    # Add information labels
    tk.Label(frame, text="DPI-Aware Autoscaling Test",
             font=("Arial", scale_value(16), "bold")).pack(pady=scale_value(10))

    tk.Label(frame, text=scale_info, font=("Arial", scale_value(12))).pack(pady=scale_value(5))
    tk.Label(frame, text=dpi_info, font=("Arial", scale_value(12))).pack(pady=scale_value(5))

    # Test button with scaled font
    tk.Button(frame, text="Test Button",
              font=("Arial", scale_value(12)),
              command=lambda: print("Button clicked!")).pack(pady=scale_value(20))

    # Test label with scaled wraplength
    test_text = "This is a test of text wrapping with DPI-aware scaling. " * 3
    tk.Label(frame, text=test_text,
             wraplength=scale_value(400),
             justify="left").pack(pady=scale_value(10))

    # Test image placeholder with scaled size
    canvas = tk.Canvas(frame, width=scale_value(200), height=scale_value(150), bg="lightgray")
    canvas.pack(pady=scale_value(10))
    canvas.create_text(scale_value(100), scale_value(75),
                      text=f"Scaled Canvas\n{scale_value(200)}x{scale_value(150)}",
                      font=("Arial", scale_value(10)))

    # Exit button
    tk.Button(frame, text="Exit",
              font=("Arial", scale_value(12)),
              command=root.destroy).pack(pady=scale_value(20))

    root.mainloop()


if __name__ == '__main__':
    main()

