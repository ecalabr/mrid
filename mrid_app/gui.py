import os
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from . import config
from .config import DATA_DIR, OUT_DIR, NUM_COLUMNS, NUM_ROWS, ALLOW_DUPLICATE_LABELS, NUM_PARALLEL_WORKERS
from .header import TrainHeaderClassifierLogic
from .cnn import SimpleCNN
from .utils import (extract_study_id_from_path, get_study_csv_name, get_study_dir_name,
                     get_nifti_name, get_nifti_prefix, sanitize_target_series)
from .logging_config import get_logger, setup_logging, update_log_file_location
import joblib
from PIL import Image, ImageTk
import shutil
import colorsys
import platform
import subprocess
import importlib

logger = get_logger()


# Sanitize TARGET_SERIES to ensure safe filenames
TARGET_SERIES = sanitize_target_series(config.TARGET_SERIES)


# Global scaling factor to be calculated once
_SCALE_FACTOR = None


def _calculate_scale_factor(window):
    """Calculate the appropriate scaling factor based on DPI and platform."""
    try:
        # Get DPI from the window
        dpi = window.winfo_fpixels('1i')

        # Base DPI reference (standard 96 DPI for Windows, 72 for Mac)
        if platform.system() == 'Darwin':  # macOS
            base_dpi = 72.0
        else:  # Windows/Linux
            base_dpi = 96.0

        # Calculate scaling factor
        scale = dpi / base_dpi

        # Clamp the scaling factor to reasonable bounds
        scale = max(0.75, min(scale, 3.0))

        return scale
    except Exception:
        return 1.0


def _enable_dpi_awareness():
    """Enable DPI awareness for Windows to get correct DPI values."""
    if platform.system() == 'Windows':
        try:
            from ctypes import windll
            # Try SetProcessDpiAwareness first (Windows 8.1+)
            try:
                windll.shcore.SetProcessDpiAwareness(1)
            except Exception:
                # Fall back to SetProcessDPIAware (Windows Vista+)
                try:
                    windll.user32.SetProcessDPIAware()
                except Exception:
                    pass
        except Exception:
            pass


def enable_tkinter_autoscaling(window):
    """
    Enable DPI-aware autoscaling for tkinter windows.
    Call this once on the root window, and once on each Toplevel window.
    """
    global _SCALE_FACTOR

    # Enable DPI awareness on Windows (only needs to be done once)
    if _SCALE_FACTOR is None:
        _enable_dpi_awareness()

    # Calculate scale factor if not already done
    if _SCALE_FACTOR is None:
        _SCALE_FACTOR = _calculate_scale_factor(window)

    # Apply tk scaling
    try:
        window.tk.call('tk', 'scaling', _SCALE_FACTOR * 1.33)
    except Exception:
        pass

    # Scale fonts
    try:
        import tkinter.font as tkfont

        # Default font sizes at 96 DPI
        default_sizes = {
            "TkDefaultFont": 9,
            "TkTextFont": 9,
            "TkFixedFont": 9,
            "TkMenuFont": 9,
            "TkHeadingFont": 9,
            "TkCaptionFont": 9,
            "TkSmallCaptionFont": 8,
            "TkIconFont": 9,
            "TkTooltipFont": 8,
        }

        for name, default_size in default_sizes.items():
            try:
                font = tkfont.nametofont(name)
                # Scale the default size, not the current size
                new_size = int(default_size * _SCALE_FACTOR)
                # Ensure minimum readable size
                new_size = max(8, new_size)
                font.configure(size=new_size)
            except Exception:
                continue
    except Exception:
        pass


def scale_geometry(geometry_string):
    """
    Scale a geometry string (e.g., "500x300") by the global scale factor.
    Returns the scaled geometry string.
    """
    global _SCALE_FACTOR
    if _SCALE_FACTOR is None:
        return geometry_string

    try:
        # Parse geometry string like "500x300" or "500x300+100+100"
        parts = geometry_string.split('+')
        size_part = parts[0]
        position_parts = parts[1:] if len(parts) > 1 else []

        width, height = map(int, size_part.split('x'))
        scaled_width = int(width * _SCALE_FACTOR)
        scaled_height = int(height * _SCALE_FACTOR)

        result = f"{scaled_width}x{scaled_height}"
        if position_parts:
            result += '+' + '+'.join(position_parts)

        return result
    except Exception:
        return geometry_string


def scale_value(value):
    """Scale a numeric value by the global scale factor."""
    global _SCALE_FACTOR
    if _SCALE_FACTOR is None:
        return value
    return int(value * _SCALE_FACTOR)


def _generate_color_map(labels):
    """Generate high-contrast colors for each label.
    Returns dict label -> (bg_hex, fg_hex).
    """
    cmap = {}
    n = max(1, len(labels))
    for i, lab in enumerate(labels):
        h = float(i) / n
        s = 0.7
        v = 0.92
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        bh = '#%02x%02x%02x' % (int(r * 255), int(g * 255), int(b * 255))
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        fh = '#000000' if lum > 0.5 else '#ffffff'
        cmap[lab] = (bh, fh)
    return cmap


class TrainClassifierGUI:
    def __init__(self, master):
        self.root = tk.Toplevel(master)
        enable_tkinter_autoscaling(self.root)
        self.root.title("Model Training In Progress...")
        self.root.geometry(scale_geometry("450x120"))
        self.root.protocol("WM_DELETE_WINDOW", self.do_nothing)
        self.root.grab_set()
        self.label = tk.StringVar(value="Initializing training process...")
        tk.Label(self.root, textvariable=self.label).pack(pady=scale_value(10))
        self.progress = ttk.Progressbar(self.root, orient="horizontal",
                                        length=scale_value(400), mode="indeterminate")
        self.progress.pack(pady=scale_value(5), padx=scale_value(20))
        self.progress.start(10)
        self.root.update()

    def update_status(self, new_text):
        self.root.after(0, lambda: self.label.set(new_text))
        self.root.after(0, self.root.update_idletasks)

    def set_determinate_mode(self, maximum):
        """Switch progress bar to determinate mode with a maximum value"""
        def _update():
            self.progress.stop()
            self.progress.config(mode="determinate", maximum=maximum, value=0)
            self.root.update_idletasks()
        self.root.after(0, _update)

    def set_indeterminate_mode(self):
        """Switch progress bar back to indeterminate mode"""
        def _update():
            self.progress.config(mode="indeterminate")
            self.progress.start(10)
            self.root.update_idletasks()
        self.root.after(0, _update)

    def update_progress(self, value):
        """Update progress bar value (for determinate mode)"""
        def _update():
            self.progress["value"] = value
            self.root.update_idletasks()
        self.root.after(0, _update)

    def close(self):
        def _safe_close():
            # stop progressbar if it exists
            try:
                if hasattr(self, 'progress') and self.progress.winfo_exists():
                    try:
                        self.progress.stop()
                    except Exception:
                        pass
            except Exception:
                pass
            # release grab and destroy root if still present
            try:
                if hasattr(self, 'root') and self.root.winfo_exists():
                    try:
                        self.root.grab_release()
                    except Exception:
                        pass
                    try:
                        self.root.destroy()
                    except Exception:
                        pass
            except Exception:
                pass

        try:
            self.root.after(0, _safe_close)
        except Exception:
            # fallback: call synchronously
            _safe_close()

    def do_nothing(self):
        pass


class ProgressGUI:
    def __init__(self, total_files, title="Scanning DICOM Files...", label_text="Scanning files..."):
        try:
            self.root = tk.Toplevel(tk._default_root)
        except AttributeError:
            self.root = tk.Toplevel()
        enable_tkinter_autoscaling(self.root)
        try:
            self.root.title(title)
        except Exception:
            pass
        self.root.geometry(scale_geometry("500x320"))
        self.root.protocol("WM_DELETE_WINDOW", self.do_nothing)

        # First progress bar: Scanning files
        self.label = tk.Label(self.root, text=label_text)
        self.label.pack(pady=scale_value(5))
        self.progress = ttk.Progressbar(self.root, orient="horizontal",
                                        length=scale_value(400), mode="determinate",
                                        maximum=total_files)
        self.progress.pack(pady=scale_value(3))
        self.count_label = tk.Label(self.root, text="0 / 0")
        self.count_label.pack(pady=scale_value(3))

        # Second progress bar: Organizing series (initially hidden)
        self.organize_label = tk.Label(self.root, text="")
        self.organize_progress = ttk.Progressbar(self.root, orient="horizontal",
                                                  length=scale_value(400), mode="determinate",
                                                  maximum=100)
        self.organize_count_label = tk.Label(self.root, text="")

        # Third progress bar: Generating thumbnails (initially hidden)
        self.thumbnail_label = tk.Label(self.root, text="")
        self.thumbnail_progress = ttk.Progressbar(self.root, orient="horizontal",
                                                  length=scale_value(400), mode="determinate",
                                                  maximum=100)
        self.thumbnail_count_label = tk.Label(self.root, text="")

        self.root.update()

    def update(self, value, total=None):
        """Update first progress bar with counter text."""
        self.progress["value"] = value
        if total is None:
            total = self.progress["maximum"]
        self.count_label.config(text=f"{value} / {int(total)}")
        self.root.update_idletasks()

    def show_organize_progress(self, label_text="Organizing series...", total=1):
        """Show the second progress bar for organizing series."""
        self.organize_label.config(text=label_text)
        self.organize_label.pack(pady=scale_value(5))
        self.organize_progress["maximum"] = total
        self.organize_progress.pack(pady=scale_value(3))
        self.organize_count_label.pack(pady=scale_value(3))
        self.root.update_idletasks()

    def update_organize_progress(self, current, total):
        """Update organizing progress bar and count text."""
        self.organize_progress["maximum"] = total
        self.organize_progress["value"] = current
        self.organize_count_label.config(text=f"{current} / {total}")
        self.root.update_idletasks()

    def show_thumbnail_progress(self, label_text="Generating thumbnails...", total=1):
        """Show the third progress bar for thumbnail generation."""
        self.thumbnail_label.config(text=label_text)
        self.thumbnail_label.pack(pady=scale_value(5))
        self.thumbnail_progress["maximum"] = total
        self.thumbnail_progress.pack(pady=scale_value(3))
        self.thumbnail_count_label.pack(pady=scale_value(3))
        self.root.update_idletasks()

    def update_thumbnail_progress(self, current, total):
        """Update thumbnail progress bar and count text."""
        self.thumbnail_progress["maximum"] = total
        self.thumbnail_progress["value"] = current
        self.thumbnail_count_label.config(text=f"{current} / {total}")
        self.root.update_idletasks()

    def close(self):
        self.root.destroy()

    def do_nothing(self):
        pass


class MainApp:
    def __init__(self, master):
        self.master = master
        master.title("DICOM Classification Tool")
        enable_tkinter_autoscaling(master)
        self.input_dir = tk.StringVar(value=DATA_DIR)
        self.output_dir = tk.StringVar(value=OUT_DIR)
        self.classification_mode = tk.StringVar(value="Manual")
        self.model_type = tk.StringVar(value="DICOM Header Model")
        self.rf_model = None
        self.rf_target_encoder = None
        self.rf_feature_encoder = None
        self.rf_field_types = None  # Store field types for consistent preprocessing
        self.cnn_model = None
        self.cnn_target_encoder = None
        self.stop_requested = threading.Event()
        self.is_running = False
        # New flag for nifti conversion
        self.convert_to_nifti = tk.BooleanVar(value=False)
        # Flag to indicate models should be loaded after training completes
        self._auto_load_after_training = False

        # Create menu bar
        menubar = tk.Menu(master)
        master.config(menu=menubar)

        # Other Options menu
        other_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File Options", menu=other_menu)
        other_menu.add_command(label="Convert NIfTIs", command=self._convert_niftis_utility)
        other_menu.add_command(label="Erase Data", command=self._erase_data_utility)

        # Configuration menu
        config_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Configuration", menu=config_menu)
        config_menu.add_command(label="View Configuration", command=self._view_configuration_dialog)

        # UI layout
        dir_frame = tk.LabelFrame(master, text="Directories", padx=scale_value(10), pady=scale_value(10))
        dir_frame.pack(padx=scale_value(20), pady=scale_value(10), fill="x")
        tk.Label(dir_frame, text="Input DIR:").grid(row=0, column=0, padx=scale_value(5), pady=scale_value(5), sticky="w")
        tk.Entry(dir_frame, textvariable=self.input_dir, width=50).grid(row=0, column=1, padx=scale_value(5), pady=scale_value(5), sticky="ew")
        tk.Button(dir_frame, text="Browse", command=lambda: self.browse_dir(self.input_dir, "Select INPUT directory")).grid(row=0, column=2, padx=scale_value(5), pady=scale_value(5))
        tk.Label(dir_frame, text="Output DIR:").grid(row=1, column=0, padx=scale_value(5), pady=scale_value(5), sticky="w")
        tk.Entry(dir_frame, textvariable=self.output_dir, width=50).grid(row=1, column=1, padx=scale_value(5), pady=scale_value(5), sticky="ew")
        tk.Button(dir_frame, text="Browse", command=lambda: self.browse_dir(self.output_dir, "Select OUTPUT directory")).grid(row=1, column=2, padx=scale_value(5), pady=scale_value(5))
        dir_frame.columnconfigure(1, weight=1)

        control_frame = tk.Frame(master, padx=scale_value(20), pady=scale_value(10))
        control_frame.pack(fill="x")
        mode_frame = tk.Frame(control_frame)
        mode_frame.pack(side="right", padx=scale_value(5))
        tk.Label(mode_frame, text="Mode:").pack(side="left", padx=scale_value(5))
        mode_options = ["Manual", "Semi-Auto", "Full-Auto", "Verify"]
        # Replaced ttk.Combobox with tk.OptionMenu. Keep using the same
        # StringVar (self.classification_mode) and call _check_model_status via
        # the OptionMenu 'command' so existing logic that relies on the
        # variable and event handling remains unchanged.
        mode_menu = tk.OptionMenu(mode_frame, self.classification_mode, *mode_options, command=self._check_model_status)
        try:
            mode_menu.config(width=12)
        except Exception:
            pass
        mode_menu.pack(side="left", padx=scale_value(5))
        model_type_frame = tk.Frame(control_frame)
        model_type_frame.pack(side="right", padx=scale_value(5))
        tk.Label(model_type_frame, text="Model Type:").pack(side="left", padx=scale_value(5))
        model_options = ["DICOM Header Model", "Image-Based Model", "Header + Image Model"]
        # Replace model-type Combobox with OptionMenu; preserve self.model_type
        # StringVar and call _check_model_status on selection changes.
        model_menu = tk.OptionMenu(model_type_frame, self.model_type, *model_options, command=self._check_model_status)
        try:
            model_menu.config(width=20)
        except Exception:
            pass
        model_menu.pack(side="left", padx=scale_value(5))

        # New convert toggle (Yes/No)
        convert_frame = tk.Frame(control_frame)
        convert_frame.pack(side="right", padx=scale_value(10))
        tk.Label(convert_frame, text="Convert to NIfTI:").pack(side="left", padx=(0, scale_value(5)))

        # Single Button toggle showing Yes/No. Clicking flips the underlying BooleanVar
        def _convert_button_toggle():
            new_state = not bool(self.convert_to_nifti.get())
            self.convert_to_nifti.set(new_state)
            # call the existing handler which performs dcm2niix check and may revert
            self._on_toggle_convert()
            # reflect final state on the button text
            try:
                self._convert_button.config(text=("Yes" if self.convert_to_nifti.get() else "No"))
            except Exception:
                pass

        self._convert_button = tk.Button(convert_frame, text=("Yes" if self.convert_to_nifti.get() else "No"), command=_convert_button_toggle, bd=1, relief='raised')
        self._convert_button.pack(side="left")

        self.start_button = tk.Button(control_frame, text="▶ Start Classification", font=("Arial", scale_value(12), "bold"), command=self.start_classification, state="normal")
        self.start_button.pack(side="left", padx=scale_value(5))
        self.stop_button = tk.Button(control_frame, text="Stop", font=("Arial", scale_value(12)), command=self.stop_classification, state="disabled")
        self.stop_button.pack(side="left", padx=scale_value(5))
        self.train_button = tk.Button(control_frame, text="Train Model(s)", font=("Arial", scale_value(12)), command=self._train_model)
        self.train_button.pack(side="right", padx=scale_value(5))
        self.load_button = tk.Button(control_frame, text="Load Model(s)", font=("Arial", scale_value(12)), command=self._load_model_handler)
        self.load_button.pack(side="right", padx=scale_value(5))
        self._check_model_status()

    def browse_dir(self, var, title):
        directory = filedialog.askdirectory(title=title, initialdir=var.get())
        if directory:
            var.set(directory)

    def _on_toggle_convert(self):
        """Handler for the convert-to-nifti toggle. If enabling, check for dcm2niix in PATH.
        If not found, revert toggle and show an error dialog."""
        enabled = bool(self.convert_to_nifti.get())
        # update the button text to show current state
        try:
            # update the main toggle button to reflect the state
            if hasattr(self, '_convert_button'):
                self._convert_button.config(text=("Yes" if enabled else "No"))
        except Exception:
            pass

        if enabled:
            # Check for dcm2niix availability
            dcm2niix_path = shutil.which("dcm2niix")
            if not dcm2niix_path:
                try:
                    messagebox.showerror("Missing dcm2niix", "dcm2niix was not found in your PATH. Please install it or ensure it's available in PATH.", parent=self.master)
                except Exception:
                    logger.error("dcm2niix was not found in PATH. Please install or add to PATH.")
                # revert toggle
                self.convert_to_nifti.set(False)
                try:
                    if hasattr(self, '_convert_button'):
                        self._convert_button.config(text="No")
                except Exception:
                    pass
                return

    def _load_model_handler(self):
        """Handle Load Model(s) button press. If models don't exist, ask user if they want to train them."""
        output_dir = self.output_dir.get()
        model_type = self.model_type.get()
        rf_needed = model_type in ["DICOM Header Model", "Header + Image Model"]
        cnn_needed = model_type in ["Image-Based Model", "Header + Image Model"]

        rf_model_path = os.path.join(output_dir, "rf_classifier_model.joblib")
        cnn_model_path = os.path.join(output_dir, "cnn_classifier_model.pth")

        # Check which models exist and which are missing
        rf_exists = os.path.exists(rf_model_path) if rf_needed else True
        cnn_exists = os.path.exists(cnn_model_path) if cnn_needed else True

        # If all needed models exist, load them
        if rf_exists and cnn_exists:
            if rf_needed:
                self._load_classifier_and_encoder()
            if cnn_needed:
                self._load_image_classifier()
            self._check_model_status()
            return

        # Models are missing - ask user if they want to train
        missing_models = []
        if rf_needed and not rf_exists:
            missing_models.append("DICOM Header Model")
        if cnn_needed and not cnn_exists:
            missing_models.append("Image-Based Model")

        missing_str = " and ".join(missing_models)
        response = messagebox.askyesno(
            "Models Not Found",
            f"The following model(s) are not trained yet:\n\n{missing_str}\n\n"
            f"Would you like to train them now?\n\n"
            f"Note: Models will be automatically loaded after training completes.",
            parent=self.master
        )

        if not response:
            return

        # User wants to train - set flag to load models after training
        self._auto_load_after_training = True

        # Start training for missing models
        if rf_needed and not rf_exists:
            self.train_classifier_rf()
        if cnn_needed and not cnn_exists:
            self.train_classifier_cnn()

    def _load_classifier_and_encoder(self):
        output_dir = self.output_dir.get()
        model_path = os.path.join(output_dir, "rf_classifier_model.joblib")
        target_encoder_path = os.path.join(output_dir, "label_encoder.joblib")
        feature_encoder_path = os.path.join(output_dir, "feature_encoder.joblib")
        field_types_path = os.path.join(output_dir, "field_types.joblib")

        if not os.path.exists(model_path) or not os.path.exists(target_encoder_path) or not os.path.exists(feature_encoder_path):
            logger.info("Header Model files not found - skipping RF model load")
            messagebox.showwarning("Load Warning", "Header Model files not found. Skipping RF load.", parent=self.master)
            return
        try:
            logger.info(f"Loading DICOM Header Model from {output_dir}")
            self.rf_model = joblib.load(model_path)
            self.rf_target_encoder = joblib.load(target_encoder_path)
            self.rf_feature_encoder = joblib.load(feature_encoder_path)

            # Load field types if available for consistent preprocessing
            if os.path.exists(field_types_path):
                self.rf_field_types = joblib.load(field_types_path)
                logger.info("Loaded field types for consistent preprocessing")
            else:
                self.rf_field_types = None
                logger.info("Field types not found - preprocessing will use automatic type detection")

            logger.info("DICOM Header Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load RF model/encoders: {e}")
            messagebox.showerror("Load Error", f"Failed to load RF model/encoders: {e}", parent=self.master)
            self.rf_model = None
            self.rf_target_encoder = None
            self.rf_feature_encoder = None
            self.rf_field_types = None

    def _load_image_classifier(self):
        try:
            import torch
            if torch is None:
                logger.info("PyTorch not installed - skipping CNN model load")
                messagebox.showwarning("Load Warning", "PyTorch is not installed. Skipping CNN load.", parent=self.master)
                return
        except Exception:
            logger.info("PyTorch not installed - skipping CNN model load")
            messagebox.showwarning("Load Warning", "PyTorch is not installed. Skipping CNN load.", parent=self.master)
            return
        output_dir = self.output_dir.get()
        model_path = os.path.join(output_dir, "cnn_classifier_model.pth")
        target_encoder_path = os.path.join(output_dir, "label_encoder.joblib")
        if not os.path.exists(model_path) or not os.path.exists(target_encoder_path):
            logger.info("Image Model files not found - skipping CNN model load")
            messagebox.showwarning("Load Warning", "Image Model files not found. Skipping CNN load.", parent=self.master)
            return
        try:
            logger.info(f"Loading Image-Based Model (CNN) from {output_dir}")
            from .utils import warn_if_no_gpu_for_cnn
            # Show a warning/confirmation dialog if no GPU is available. If the user chooses not
            # to proceed (or PyTorch missing), skip loading.
            if not warn_if_no_gpu_for_cnn(gui_parent=self.master):
                logger.info("User chose not to load CNN model without GPU")
                return

            self.cnn_target_encoder = joblib.load(target_encoder_path)
            num_classes = len(self.cnn_target_encoder.classes_)
            self.cnn_model = SimpleCNN(num_classes=num_classes)

            # Load saved state dict with map_location to handle CPU-only environments
            try:
                cuda_avail = torch.cuda.is_available()
            except Exception:
                cuda_avail = False

            map_loc = torch.device('cuda') if cuda_avail else torch.device('cpu')
            logger.info(f"Loading CNN model on device: {map_loc}")
            state_dict = torch.load(model_path, map_location=map_loc)
            self.cnn_model.load_state_dict(state_dict)
            # Move model to the appropriate device
            try:
                self.cnn_model.to(map_loc)
            except Exception:
                pass
            self.cnn_model.eval()
            logger.info("Image-Based Model (CNN) loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load CNN model: {e}")
            messagebox.showerror("Load Error", f"Failed to load CNN model: {e}", parent=self.master)
            self.cnn_model = None
            self.cnn_target_encoder = None

    def _check_model_status(self, event=None):
        model_type = self.model_type.get()
        mode = self.classification_mode.get()
        rf_loaded = self.rf_model is not None and self.rf_target_encoder is not None
        cnn_loaded = self.cnn_model is not None and self.cnn_target_encoder is not None
        can_start = False
        models_loaded = False
        mode_is_manual = False
        mode_is_verify = False

        if mode == "Manual":
            can_start = True
            mode_is_manual = True
        elif mode == "Verify":
            can_start = True
            mode_is_verify = True
        elif model_type == "DICOM Header Model" and rf_loaded:
            can_start = True
            models_loaded = True
            mode_is_manual = False
        elif model_type == "Image-Based Model" and cnn_loaded:
            can_start = True
            models_loaded = True
            mode_is_manual = False
        elif model_type == "Header + Image Model" and rf_loaded and cnn_loaded:
            can_start = True
            models_loaded = True
            mode_is_manual = False

        # Update start button
        if can_start:
            if mode_is_verify:
                self.start_button.config(text="▶ Start Verification", state="normal", bg="#4fff5b", activebackground="#4fff5b")
            else:
                self.start_button.config(text="▶ Start Classification", state="normal", bg="#4fff5b", activebackground="#4fff5b")
        else:
            self.start_button.config(text="▶ Start Classification", state="disabled", bg="#f0f0f0", activebackground="#f0f0f0")

        # Update load button
        if models_loaded:
            self.load_button.config(text="Model(s) Loaded", state="disabled", bg="#4fff5b", activebackground="#4fff5b")
        elif mode_is_manual or mode_is_verify:
            self.load_button.config(text="Load Model(s)", state="disabled", bg="#f0f0f0", activebackground="#f0f0f0")
        else:
            self.load_button.config(text="Load Model(s)", state="normal", bg="#ff4f4f", activebackground="#ff4f4f")

        train_available = True
        self.train_button.config(state="normal" if train_available else "disabled")

    def _train_model(self):
        if self.is_running:
            messagebox.showwarning("Busy", "Cannot start training while classification is running.", parent=self.master)
            return
        model_type = self.model_type.get()
        rf_needed = model_type in ["DICOM Header Model", "Header + Image Model"]
        cnn_needed = model_type in ["Image-Based Model", "Header + Image Model"]
        if not rf_needed and not cnn_needed:
            messagebox.showinfo("Info", "No model type selected for training. Please choose a Model Type.", parent=self.master)
            return
        if rf_needed:
            self.train_classifier_rf()
        if cnn_needed:
            self.train_classifier_cnn()

    def train_classifier_rf(self):
        if self.is_running: return
        output_dir = self.output_dir.get()
        try:
            import pandas as pd
            from sklearn.ensemble import RandomForestClassifier
            import joblib
            from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
        except Exception:
            logger.error("RF training dependencies not found (pandas, scikit-learn, joblib)")
            messagebox.showerror("Error", "RF training requires pandas, scikit-learn, and joblib.", parent=self.master)
            return
        if not os.path.isdir(output_dir):
            logger.error(f"Invalid output directory for training: {output_dir}")
            messagebox.showerror("Error", "Please specify a valid output directory containing classified data.", parent=self.master)
            return

        logger.info("Starting DICOM Header Model training")

        # Create callback for missing fields prompt
        def prompt_update_missing_fields(num_csvs, missing_fields):
            """Prompt user about updating CSVs with missing fields."""
            fields_str = "\n  - ".join(missing_fields)
            message = (
                f"Found {num_csvs} classification CSV(s) that are missing "
                f"the following DICOM header field(s):\n\n  - {fields_str}\n\n"
                f"Would you like to extract these fields from the DICOM files "
                f"and update the CSVs before training?\n\n"
                f"This will ensure the model uses all configured fields."
            )
            response = messagebox.askyesno(
                "Missing DICOM Header Fields",
                message,
                parent=self.master
            )
            return response

        self.start_button.config(state="disabled")
        self.train_button.config(state="disabled")
        training_gui = TrainClassifierGUI(self.master)
        training_thread = threading.Thread(
            target=TrainHeaderClassifierLogic,
            args=(output_dir, training_gui, prompt_update_missing_fields),
            daemon=True
        )
        training_thread.start()
        self.master.after(100, self._poll_training_status, training_thread)

    def train_classifier_cnn(self):
        if self.is_running: return
        output_dir = self.output_dir.get()
        try:
            import torch
        except Exception:
            logger.error("CNN training dependency not found (PyTorch)")
            messagebox.showerror("Error", "CNN training requires PyTorch (torch).", parent=self.master)
            return
        logger.info("Starting Image-Based Model (CNN) training")
        self.start_button.config(state="disabled")
        self.train_button.config(state="disabled")
        training_gui = TrainClassifierGUI(self.master)
        training_thread = threading.Thread(
            target=self._train_cnn_background,
            args=(output_dir, training_gui),
            daemon=True
        )
        training_thread.start()
        self.master.after(100, self._poll_training_status, training_thread)

    def _train_cnn_background(self, output_dir, training_gui):
        # Delegate the full image-model training flow to the cnn module's TrainImageClassifierLogic
        try:
            from . import cnn as cnn_module
            cnn_module.TrainImageClassifierLogic(output_dir, training_gui)
        finally:
            # Ensure GUI is closed if the training function returns
            try:
                training_gui.close()
            except Exception:
                pass

    def _poll_training_status(self, thread):
        if thread.is_alive():
            self.master.after(100, self._poll_training_status, thread)
        else:
            # Training completed
            self.start_button.config(state="normal")

            # If auto-load flag is set, load the models
            if self._auto_load_after_training:
                self._auto_load_after_training = False  # Reset flag

                # Load the models based on model type
                model_type = self.model_type.get()
                rf_needed = model_type in ["DICOM Header Model", "Header + Image Model"]
                cnn_needed = model_type in ["Image-Based Model", "Header + Image Model"]

                output_dir = self.output_dir.get()
                rf_model_path = os.path.join(output_dir, "rf_classifier_model.joblib")
                cnn_model_path = os.path.join(output_dir, "cnn_classifier_model.pth")

                # Load RF model if needed and exists
                if rf_needed and os.path.exists(rf_model_path):
                    self._load_classifier_and_encoder()

                # Load CNN model if needed and exists
                if cnn_needed and os.path.exists(cnn_model_path):
                    self._load_image_classifier()

            self._check_model_status()

    def start_classification(self):
        if self.is_running: return
        current_mode = self.classification_mode.get()

        # Handle Verify mode separately
        if current_mode == "Verify":
            self.start_verification()
            return

        model_type = self.model_type.get()
        if current_mode in ["Semi-Auto", "Full-Auto"]:
            rf_needed = model_type in ["DICOM Header Model", "Header + Image Model"]
            cnn_needed = model_type in ["Image-Based Model", "Header + Image Model"]
            rf_loaded = self.rf_model is not None and self.rf_target_encoder is not None
            cnn_loaded = self.cnn_model is not None and self.cnn_target_encoder is not None
            if (rf_needed and not rf_loaded) or (cnn_needed and not cnn_loaded):
                logger.error("Cannot start classification - required model(s) not loaded")
                messagebox.showerror("Model Error", "Cannot start classification. Required model(s) are not loaded.", parent=self.master)
                return
        input_dir = self.input_dir.get()
        output_dir = self.output_dir.get()
        if not os.path.isdir(input_dir):
            logger.error(f"Invalid input directory: {input_dir}")
            messagebox.showerror("Error", "Please specify a valid input directory.", parent=self.master)
            return

        logger.info(f"Starting classification - Mode: {current_mode}, Model: {model_type}")
        logger.info(f"Input directory: {input_dir}")
        logger.info(f"Output directory: {output_dir}")

        # New: if hierarchy.yaml exists in output_dir, ask user whether to load it or rescan
        existing_hierarchy = None
        try:
            hierarchy_path = os.path.join(output_dir, "hierarchy.yaml")
            if os.path.exists(hierarchy_path):
                load_choice = messagebox.askyesno(
                    "Existing hierarchy found",
                    "A saved 'hierarchy.yaml' was found in the output directory.\n\n"
                    "Choose 'Yes' to load the existing hierarchy and skip scanning files,\n"
                    "or 'No' to rescan all files.",
                    parent=self.master)
                if load_choice:
                    try:
                        import yaml
                        with open(hierarchy_path, 'r') as hf:
                            existing_hierarchy = yaml.safe_load(hf)
                        logger.info("Loaded existing hierarchy.yaml - skipping DICOM scan")
                    except Exception as e:
                        logger.warning(f"Failed to load existing hierarchy.yaml: {e} - will rescan files")
                        messagebox.showwarning("Load Warning", f"Failed to load existing hierarchy.yaml: {e}\nWill rescan files instead.", parent=self.master)
                        existing_hierarchy = None
                else:
                    logger.info("User chose to rescan files instead of using existing hierarchy")
        except Exception:
            existing_hierarchy = None

        self.is_running = True
        self.stop_requested.clear()
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.train_button.config(state="disabled")

        self.progress_frame = tk.Frame(self.master)
        self.progress_frame.pack(padx=20, pady=10, fill='x')

        self.progress_label = tk.StringVar(value="Starting...")
        tk.Label(self.progress_frame, textvariable=self.progress_label).pack(side="top")

        self.progress_bar = ttk.Progressbar(self.progress_frame, orient="horizontal", mode="determinate")
        self.progress_bar.pack(fill='x', pady=5)

        # Pass existing_hierarchy into the background worker so it can skip scanning when appropriate
        classification_thread = threading.Thread(
            target=self.run_classification_main,
            args=(input_dir, output_dir, current_mode, existing_hierarchy),
            daemon=True
        )
        classification_thread.start()
        self.master.after(100, self._poll_classification_status, classification_thread)

    def stop_classification(self):
        if not self.is_running: return
        self.stop_requested.set()
        self.stop_button.config(text="Stopping...", state="disabled")

    def _poll_classification_status(self, thread):
        if thread.is_alive():
            self.master.after(100, self._poll_classification_status, thread)
        else:
            self.classification_finished(aborted=self.stop_requested.is_set())

    def classification_finished(self, aborted):
        self.is_running = False
        self.stop_requested.clear()
        self.start_button.config(state="normal")
        self.stop_button.config(text="Stop", state="disabled")
        self.train_button.config(state="normal")

        if aborted:
            logger.info("Classification workflow stopped by user")
        else:
            logger.info("Classification workflow completed successfully")

        if hasattr(self, 'progress_frame') and self.progress_frame.winfo_exists():
            self.progress_frame.destroy()
        self._check_model_status()

    def run_classification_main(self, input_dir, output_dir, mode, existing_hierarchy=None):
        """
        Main classification workflow, controlling execution based on the selected mode.
        This method runs in a background thread and interacts with the MainApp instance
        via instance attributes (progress_bar, stop_requested, etc.).

        If `existing_hierarchy` is provided (a dict), scanning of DICOM files is skipped and that
        hierarchy is used directly.
        """
        try:
            import multiprocessing as mp
            from .scanner import build_hierarchy
            from .utils import write_yaml_hierarchy
            from . import processing

            # Set up file logging now that we have the output directory
            # This will flush any buffered log records to the file
            os.makedirs(output_dir, exist_ok=True)
            update_log_file_location(output_dir)

            # If an existing hierarchy was supplied, use it and skip scanning
            if existing_hierarchy:
                hierarchy = existing_hierarchy
                dicom_records = []
            else:
                # 1. Collect all files
                all_files = []
                for dcm_root, _, files in os.walk(input_dir):
                    for f in files:
                        if not f.startswith(".") and os.path.splitext(f)[1] in ["", ".dcm", ".dicom"]:
                            all_files.append(os.path.join(dcm_root, f))

                logger.info(f"Found {len(all_files)} files. Scanning for DICOM...")

                # 2. Setup progress window
                total_files = len(all_files)
                gui = ProgressGUI(total_files=total_files)

                # 3. Parallel processing (scan) — use batched workers to reduce per-file overhead
                pool = mp.Pool(mp.cpu_count())
                results = []

                try:
                    # Create chunks of files to process per worker call (batching reduces per-file overhead)
                    batch_size = 64
                    chunks = [all_files[i:i+batch_size] for i in range(0, len(all_files), batch_size)]

                    # Use the batched scanner; import the batch function
                    from .scanner import check_dicom_batch

                    processed = 0
                    for batch_res in pool.imap_unordered(check_dicom_batch, chunks):
                        # batch_res is a list of per-file results (or None entries)
                        if batch_res:
                            results.extend(batch_res)
                            processed += len(batch_res)
                        else:
                            # if a worker returned None or failed, assume the corresponding chunk files yielded None
                            processed += batch_size
                        # clamp processed to total files
                        if processed > total_files:
                            processed = total_files
                        gui.update(processed, total_files)

                        if self.stop_requested.is_set():
                            gui.close()
                            pool.terminate()
                            pool.join()
                            self.classification_finished(aborted=True)
                            return
                finally:
                    pool.close()
                    pool.join()

                # 4. Process results
                dicom_records = [r for r in results if r is not None]
                logger.info(f"Identified {len(dicom_records)} valid DICOM files.")

                # Count unique groups for hierarchy building
                try:
                    groups = set((r['patient'], r['study_uid'], r['series_uid']) for r in dicom_records)
                    total_groups = max(1, len(groups))
                except Exception:
                    total_groups = max(1, len(dicom_records))

                # Reuse the same GUI window for hierarchy building and thumbnail generation
                organize_started = False
                thumbnail_started = False
                try:
                    # Callback for hierarchy building progress
                    def _build_progress_cb(n):
                        nonlocal organize_started
                        try:
                            if gui:
                                if not organize_started:
                                    gui.show_organize_progress("Organizing series...", total_groups)
                                    organize_started = True
                                gui.update_organize_progress(n, total_groups)
                        except Exception:
                            pass

                    # Callback for thumbnail generation progress
                    def _thumbnail_progress_cb(current, total):
                        nonlocal thumbnail_started
                        try:
                            if gui:
                                if not thumbnail_started:
                                    gui.show_thumbnail_progress("Generating thumbnails...", total)
                                    thumbnail_started = True
                                gui.update_thumbnail_progress(current, total)
                        except Exception:
                            pass

                    hierarchy = build_hierarchy(
                        dicom_records,
                        output_dir,
                        progress_callback=_build_progress_cb,
                        thumbnail_progress_callback=_thumbnail_progress_cb
                    )
                    # save hierarchy to YAML
                    try:
                        write_yaml_hierarchy(hierarchy, output_dir)
                    except Exception as e:
                        logger.warning(f"Failed to write hierarchy.yaml: {e}")
                finally:
                    try:
                        if gui:
                            gui.close()
                    except Exception:
                        pass

            # At this point `hierarchy` is available (either loaded or newly constructed)
            # 5. Loop through each study and classify
            all_studies = []
            for mrn, studies in hierarchy.items():
                for study_uid, study_data in studies.items():
                    # Extract accession number and series from the hierarchy structure
                    accn = study_data.get("accn", "")
                    series_dict = study_data.get("series", {})
                    all_studies.append((mrn, study_uid, accn, series_dict))

            total_studies = len(all_studies)
            try:
                self.progress_bar.config(maximum=total_studies)
            except Exception:
                pass

            model_type = self.model_type.get()
            rf_model, rf_target_encoder, rf_feature_encoder = self.rf_model, self.rf_target_encoder, self.rf_feature_encoder
            cnn_model, cnn_target_encoder = self.cnn_model, self.cnn_target_encoder

            for i, (mrn, study_uid, accn, series_dict) in enumerate(all_studies):
                if self.stop_requested.is_set():
                    logger.info("Classification loop aborted by user.")
                    self.classification_finished(aborted=True)
                    return


                self.progress_label.set(
                    f"Classifying Study {i + 1} of {total_studies} ({mode}/{model_type}): {mrn} / {study_uid}")

                study_dir_name = get_study_dir_name(mrn, study_uid, accn)
                classification_csv = os.path.join(output_dir, study_dir_name, get_study_csv_name(study_dir_name))
                if os.path.exists(classification_csv):
                    try:
                        self.progress_bar.config(value=i + 1)
                        self.master.update_idletasks()
                    except Exception:
                        pass
                    continue

                if mode == "Full-Auto":
                    processing.full_auto_classify_study(
                        series_dict, mrn, study_uid, output_dir,
                        model_type, rf_model, rf_target_encoder, rf_feature_encoder, cnn_model, cnn_target_encoder,
                        self.rf_field_types,  # Pass field types
                        convert=self.convert_to_nifti.get(),
                        accn=accn
                    )
                else:
                    study_done_event = threading.Event()

                    # Call the per-study GUI (defined below in this module)
                    classify_study_gui(
                        series_dict, mrn, study_uid, output_dir, self.stop_requested,
                        study_done_event, mode,
                        model_type, rf_model, rf_target_encoder, rf_feature_encoder, cnn_model, cnn_target_encoder,
                        self.rf_field_types,  # Pass field types
                        convert=self.convert_to_nifti.get(),
                        accn=accn
                    )

                    while not study_done_event.is_set():
                        time.sleep(0.1)
                        if self.stop_requested.is_set():
                            self.classification_finished(aborted=True)
                            return

                try:
                    self.progress_bar.config(value=i + 1)
                    self.master.update_idletasks()
                except Exception:
                    pass

            self.classification_finished(aborted=False)

        except Exception as e:
            logger.exception(f"An error occurred in the classification process: {e}")
            self.classification_finished(aborted=True)

    def _convert_niftis_utility(self):
        """Utility to convert all classified studies to NIfTI format."""
        if self.is_running:
            messagebox.showwarning("Busy", "Cannot convert while classification is running.", parent=self.master)
            return

        output_dir = self.output_dir.get()
        if not os.path.isdir(output_dir):
            logger.error(f"Invalid output directory for NIfTI conversion: {output_dir}")
            messagebox.showerror("Error", "Please specify a valid output directory.", parent=self.master)
            return

        # Ask user for confirmation
        response = messagebox.askyesno(
            "Convert NIfTIs",
            "This will convert all classified studies to NIfTI format (if not already converted).\n\n"
            "This may take a while depending on the number of studies.\n\n"
            "Continue?",
            parent=self.master
        )

        if not response:
            logger.info("User cancelled NIfTI conversion")
            return

        logger.info("Starting NIfTI conversion utility")

        # Check for dcm2niix
        dcm2niix_path = shutil.which("dcm2niix")
        if not dcm2niix_path:
            logger.error("dcm2niix not found in PATH")
            messagebox.showerror(
                "Missing dcm2niix",
                "dcm2niix was not found in your PATH. Please install it to convert to NIfTI.",
                parent=self.master
            )
            return

        # Count studies to process
        study_folders = []
        for folder in os.listdir(output_dir):
            folder_path = os.path.join(output_dir, folder)
            if os.path.isdir(folder_path):
                # Check if it looks like a study folder (has classification CSV)
                study_id = extract_study_id_from_path(folder_path)
                csv_path = os.path.join(folder_path, get_study_csv_name(study_id))
                if os.path.exists(csv_path):
                    study_folders.append(folder_path)

        if not study_folders:
            messagebox.showinfo("No Studies", "No classified studies found in output directory.", parent=self.master)
            return

        # Create progress window
        progress_window = tk.Toplevel(self.master)
        enable_tkinter_autoscaling(progress_window)
        progress_window.title("Converting to NIfTI...")
        progress_window.geometry(scale_geometry("500x150"))
        progress_window.protocol("WM_DELETE_WINDOW", lambda: None)  # Disable close

        status_label = tk.Label(progress_window, text=f"Converting {len(study_folders)} studies...")
        status_label.pack(pady=scale_value(10))

        progress_bar = ttk.Progressbar(progress_window, orient="horizontal", length=scale_value(400), mode="determinate", maximum=len(study_folders))
        progress_bar.pack(pady=scale_value(10))

        current_label = tk.Label(progress_window, text="")
        current_label.pack(pady=scale_value(5))

        # Convert in background thread
        def convert_worker():
            converted_count = 0
            skipped_count = 0
            error_count = 0

            for idx, study_path in enumerate(study_folders):
                study_id = extract_study_id_from_path(study_path)

                # Update UI
                progress_window.after(0, lambda i=idx, sid=study_id: current_label.config(text=f"Processing: {sid}"))
                progress_window.after(0, lambda i=idx: progress_bar.config(value=i))

                # Read classification CSV
                csv_path = os.path.join(study_path, get_study_csv_name(study_id))
                try:
                    import pandas as pd
                    df = pd.read_csv(csv_path)

                    for _, row in df.iterrows():
                        classification = str(row.get('Classification', '')).strip()
                        if classification.lower() in ['other', 'none', '']:
                            continue

                        path = row.get('Path', '')
                        if not path or not os.path.exists(path):
                            continue

                        # Check if NIfTI already exists
                        series_dir = os.path.dirname(path)
                        expected_nifti_name = get_nifti_name(study_id, classification)
                        expected_nifti_path = os.path.join(study_path, expected_nifti_name)

                        # Check if NIfTI file exists
                        if os.path.exists(expected_nifti_path):
                            skipped_count += 1
                            continue

                        # Convert to NIfTI using existing function
                        try:
                            processing.convert_series_to_nifti(
                                series_dir=series_dir,
                                output_path=study_path,
                                label=classification
                            )
                            converted_count += 1
                        except Exception as e:
                            logger.error(f"Error converting {series_dir}: {e}")
                            error_count += 1

                except Exception as e:
                    logger.error(f"Error processing {csv_path}: {e}")
                    error_count += 1

            # Update final progress
            progress_window.after(0, lambda: progress_bar.config(value=len(study_folders)))
            progress_window.after(0, lambda: current_label.config(text="Complete!"))

            # Show summary
            summary = "Conversion Complete!\n\n"
            summary += f"Converted: {converted_count}\n"
            summary += f"Skipped (already exist): {skipped_count}\n"
            if error_count > 0:
                summary += f"Errors: {error_count}"

            logger.info(f"NIfTI conversion complete - Converted: {converted_count}, Skipped: {skipped_count}, Errors: {error_count}")

            progress_window.after(0, lambda: messagebox.showinfo("Conversion Complete", summary, parent=self.master))
            progress_window.after(0, progress_window.destroy)

        thread = threading.Thread(target=convert_worker, daemon=True)
        thread.start()

    def _erase_data_utility(self):
        """Utility to erase selected data from output directory."""
        if self.is_running:
            messagebox.showwarning("Busy", "Cannot erase data while classification is running.", parent=self.master)
            return

        output_dir = self.output_dir.get()
        if not os.path.isdir(output_dir):
            messagebox.showerror("Error", "Please specify a valid output directory.", parent=self.master)
            return

        # Create dialog with checkboxes
        dialog = tk.Toplevel(self.master)
        enable_tkinter_autoscaling(dialog)
        dialog.title("Erase Data")
        dialog.geometry(scale_geometry("550x280"))
        dialog.transient(self.master)
        dialog.grab_set()

        tk.Label(dialog, text="Select data to erase:", font=("Arial", scale_value(12), "bold")).pack(pady=scale_value(10))

        # Checkboxes
        delete_classifications = tk.BooleanVar(value=False)
        delete_models = tk.BooleanVar(value=False)
        delete_hierarchy = tk.BooleanVar(value=False)

        tk.Checkbutton(
            dialog,
            text="Delete Classifications (study folders, NIfTIs, CSVs)",
            variable=delete_classifications,
            font=("Arial", scale_value(10)),
            wraplength=scale_value(500),
            justify="left"
        ).pack(anchor="w", padx=scale_value(20), pady=scale_value(5))

        tk.Checkbutton(
            dialog,
            text="Delete Models (trained models, encoders, results)",
            variable=delete_models,
            font=("Arial", scale_value(10)),
            wraplength=scale_value(500),
            justify="left"
        ).pack(anchor="w", padx=scale_value(20), pady=scale_value(5))

        tk.Checkbutton(
            dialog,
            text="Delete Hierarchy (hierarchy.yaml, dicom_symlinks)",
            variable=delete_hierarchy,
            font=("Arial", scale_value(10)),
            wraplength=scale_value(500),
            justify="left"
        ).pack(anchor="w", padx=scale_value(20), pady=scale_value(5))

        tk.Label(
            dialog,
            text="Warning: This action cannot be undone!",
            fg="red",
            font=("Arial", scale_value(10), "bold")
        ).pack(pady=scale_value(10))

        # Button frame
        button_frame = tk.Frame(dialog)
        button_frame.pack(pady=scale_value(10))

        def confirm_delete():
            if not any([delete_classifications.get(), delete_models.get(), delete_hierarchy.get()]):
                messagebox.showwarning("No Selection", "Please select at least one option to delete.", parent=dialog)
                return

            # Build confirmation message
            items = []
            if delete_classifications.get():
                items.append("- All classification study folders")
            if delete_models.get():
                items.append("- All trained models and training data")
            if delete_hierarchy.get():
                items.append("- Hierarchy and temporary files")

            confirm_msg = "Are you sure you want to DELETE:\n\n" + "\n".join(items) + "\n\nThis cannot be undone!"

            response = messagebox.askyesno("Confirm Deletion", confirm_msg, parent=dialog, icon='warning')
            if not response:
                logger.info("User cancelled data deletion")
                return

            logger.info(f"User confirmed data deletion - Classifications: {delete_classifications.get()}, Models: {delete_models.get()}, Hierarchy: {delete_hierarchy.get()}")
            dialog.destroy()
            self._perform_data_deletion(delete_classifications.get(), delete_models.get(), delete_hierarchy.get())

        tk.Button(button_frame, text="Delete", command=confirm_delete, bg="#ff4444", fg="white", font=("Arial", scale_value(10), "bold"), width=10).pack(side="left", padx=scale_value(5))
        tk.Button(button_frame, text="Cancel", command=dialog.destroy, font=("Arial", scale_value(10)), width=10).pack(side="left", padx=scale_value(5))

    def _perform_data_deletion(self, delete_classifications, delete_models, delete_hierarchy):
        """Perform the actual data deletion."""
        output_dir = self.output_dir.get()
        deleted_items = []
        errors = []

        try:
            # Delete classifications (study folders with CSVs)
            if delete_classifications:
                for folder in os.listdir(output_dir):
                    folder_path = os.path.join(output_dir, folder)
                    if os.path.isdir(folder_path):
                        study_id = extract_study_id_from_path(folder_path)
                        csv_path = os.path.join(folder_path, get_study_csv_name(study_id))

                        if os.path.exists(csv_path):
                            try:
                                shutil.rmtree(folder_path)
                                deleted_items.append(f"Study folder: {study_id}")
                            except Exception as e:
                                errors.append(f"Failed to delete {folder_path}: {e}")

            # Delete models
            if delete_models:
                model_files = [
                    "rf_classifier_model.joblib",
                    "cnn_classifier_model.pth",
                    "label_encoder.joblib",
                    "feature_encoder.joblib",
                    "field_types.joblib",
                    "classification_data.csv",
                    "rf_model_training_summary.txt",
                    "cnn_model_training_summary.txt"
                ]

                # Also check for CNN cache files
                for item in os.listdir(output_dir):
                    if item.startswith("cnn_preprocessed_data_") and item.endswith(".pkl"):
                        model_files.append(item)

                for filename in model_files:
                    filepath = os.path.join(output_dir, filename)
                    if os.path.exists(filepath):
                        try:
                            os.remove(filepath)
                            deleted_items.append(f"Model file: {filename}")
                        except Exception as e:
                            errors.append(f"Failed to delete {filename}: {e}")

                # Reset loaded models in GUI
                self.rf_model = None
                self.rf_target_encoder = None
                self.rf_feature_encoder = None
                self.rf_field_types = None
                self.cnn_model = None
                self.cnn_target_encoder = None
                self._check_model_status()

            # Delete hierarchy
            if delete_hierarchy:
                hierarchy_path = os.path.join(output_dir, "hierarchy.yaml")
                temp_files_path = os.path.join(output_dir, "dicom_symlinks")
                thumbnails_path = os.path.join(output_dir, "series_thumbnails")

                if os.path.exists(hierarchy_path):
                    try:
                        os.remove(hierarchy_path)
                        deleted_items.append("hierarchy.yaml")
                    except Exception as e:
                        errors.append(f"Failed to delete hierarchy.yaml: {e}")

                if os.path.exists(temp_files_path):
                    try:
                        shutil.rmtree(temp_files_path)
                        deleted_items.append("dicom_symlinks directory")
                    except Exception as e:
                        errors.append(f"Failed to delete dicom_symlinks: {e}")

                if os.path.exists(thumbnails_path):
                    try:
                        shutil.rmtree(thumbnails_path)
                        deleted_items.append("series_thumbnails directory")
                    except Exception as e:
                        errors.append(f"Failed to delete series_thumbnails: {e}")

            # Show results
            if deleted_items or errors:
                # Create simplified messages based on what was deleted
                messages = []
                if delete_classifications and any('Study folder' in item for item in deleted_items):
                    messages.append("Successfully deleted classification data")
                if delete_models and any('Model file' in item for item in deleted_items):
                    messages.append("Successfully deleted model data")
                if delete_hierarchy and (any('hierarchy.yaml' in item for item in deleted_items) or
                                        any('dicom_symlinks' in item for item in deleted_items) or
                                        any('series_thumbnails' in item for item in deleted_items)):
                    messages.append("Successfully deleted hierarchy data")

                result_msg = "\n".join(messages) if messages else "Deletion completed"

                logger.info(f"Data deletion complete - {len(deleted_items)} items deleted, {len(errors)} errors")

                if errors:
                    result_msg += f"\n\nWarning: {len(errors)} error(s) occurred during deletion."
                    if len(errors) <= 3:
                        result_msg += "\n" + "\n".join(errors)

                messagebox.showinfo("Deletion Complete", result_msg, parent=self.master)
            else:
                messagebox.showinfo("Nothing Deleted", "No matching files found to delete.", parent=self.master)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during deletion: {e}", parent=self.master)

    def _view_configuration_dialog(self):
        """Display configuration settings in a read-only dialog with edit and reload options."""
        dialog = tk.Toplevel(self.master)
        enable_tkinter_autoscaling(dialog)
        dialog.title("Configuration Settings")
        dialog.geometry(scale_geometry("700x600"))
        dialog.transient(self.master)

        # Header
        header_frame = tk.Frame(dialog)
        header_frame.pack(fill="x", padx=scale_value(10), pady=scale_value(10))
        tk.Label(
            header_frame,
            text="Current Configuration Settings",
            font=("Arial", scale_value(12), "bold")
        ).pack()

        # Text widget with scrollbar for configuration display
        text_frame = tk.Frame(dialog)
        text_frame.pack(fill="both", expand=True, padx=scale_value(10), pady=scale_value(5))

        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side="right", fill="y")

        text_widget = tk.Text(
            text_frame,
            wrap="none",
            yscrollcommand=scrollbar.set,
            font=("Courier", scale_value(9)),
            state="normal"
        )
        text_widget.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=text_widget.yview)

        # Get configuration values
        config_text = self._get_config_text()
        text_widget.insert("1.0", config_text)
        text_widget.config(state="disabled")

        # Button frame
        button_frame = tk.Frame(dialog)
        button_frame.pack(fill="x", padx=scale_value(10), pady=scale_value(10))

        tk.Button(
            button_frame,
            text="Edit Config File",
            command=self._edit_config_file,
            font=("Arial", scale_value(10)),
            width=15
        ).pack(side="left", padx=scale_value(5))

        tk.Button(
            button_frame,
            text="Reload Config",
            command=lambda: self._reload_config(dialog),
            font=("Arial", scale_value(10)),
            width=15
        ).pack(side="left", padx=scale_value(5))

        tk.Button(
            button_frame,
            text="Close",
            command=dialog.destroy,
            font=("Arial", scale_value(10)),
            width=15
        ).pack(side="right", padx=scale_value(5))

    def _get_config_text(self):
        """Generate formatted text of all user-configurable settings from config.py."""
        import pprint

        lines = []
        lines.append("=" * 70)
        lines.append("CONFIGURATION SETTINGS")
        lines.append("=" * 70)
        lines.append("")

        # Define configuration variables to display (user-configurable only)
        config_vars = [
            # Directories
            ("DATA_DIR", "Input directory for DICOM files"),
            ("OUT_DIR", "Output directory for processed data"),

            # Classification settings
            ("TARGET_SERIES", "Target series labels for classification"),
            ("PREFER_3D", "Prefer 3D acquisitions for each series type"),
            ("INCLUDE_TYPES", "DICOM ImageType values to include"),
            ("EXCLUDE_TYPES", "DICOM ImageType values to exclude"),
            ("IGNORE_MULTIPLE_ORIENTATIONS", "Ignore series with multiple orientations"),
            ("SEPARATE_STACKED_SERIES", "Separate stacked series into components"),
            ("MAX_SPLIT", "Maximum splits for stacked series"),

            # Classification UI options
            ("NUM_COLUMNS", "Number of columns in classification UI"),
            ("OVERRIDE_THRESHOLD", "Probability threshold for auto-classification"),
            ("ALLOW_DUPLICATE_LABELS", "Allow duplicate labels in manual classification"),

            # CNN hyperparameters
            ("IMAGE_SIZE", "Image size for CNN input"),
            ("NUM_EPOCHS", "Number of training epochs for CNN"),
            ("BATCH_SIZE", "Batch size for CNN training"),
            ("LR", "Learning rate for CNN"),

            # Random Forest hyperparameters
            ("N_ESTIMATORS", "Number of trees in Random Forest"),

            # DICOM header fields
            ("DICOM_HEADER_FIELDS", "DICOM header fields for classification"),

            # Advanced configuration
            ("CATEGORICAL_UNIQUE_THRESHOLD", "Threshold for categorical field detection"),
            ("TEXT_MIN_WORDS", "Minimum words for text field detection"),
            ("ARRAY_SIZE_THRESHOLD", "Threshold for array field detection"),
            ("SEMANTIC_MODEL_NAME", "Name of semantic model for text encoding"),
        ]

        for var_name, description in config_vars:
            try:
                value = getattr(config, var_name)
                lines.append(f"# {description}")

                # Use pprint for better formatting of complex structures
                if isinstance(value, (list, dict, tuple)):
                    # Check if it's a complex structure that needs multi-line formatting
                    # Use a shorter threshold to trigger pretty printing more often
                    str_len = len(repr(value))

                    if str_len > 70 or (isinstance(value, list) and len(value) > 3):
                        # Format complex structures with pprint for readability
                        formatted_value = pprint.pformat(value, width=66, compact=False)
                        lines.append(f"{var_name} = {formatted_value}")
                    else:
                        # Short lists/dicts can stay on one line
                        lines.append(f"{var_name} = {repr(value)}")
                else:
                    # Simple values use repr
                    lines.append(f"{var_name} = {repr(value)}")

                lines.append("")
            except AttributeError:
                lines.append(f"# {description}")
                lines.append(f"{var_name} = <not defined>")
                lines.append("")

        return "\n".join(lines)

    def _edit_config_file(self):
        """Open config.py in the system's default text editor."""
        try:
            # Get path to config.py
            config_path = os.path.join(os.path.dirname(__file__), 'config.py')

            if not os.path.exists(config_path):
                messagebox.showerror(
                    "Error",
                    f"Config file not found at:\n{config_path}",
                    parent=self.master
                )
                return

            # Open in default editor based on platform
            system = platform.system()
            if system == 'Darwin':  # macOS
                subprocess.Popen(['open', config_path])
            elif system == 'Windows':
                os.startfile(config_path)
            else:  # Linux and others
                subprocess.Popen(['xdg-open', config_path])

            messagebox.showinfo(
                "Config File Opened",
                "The config file has been opened in your default text editor.\n\n"
                "After making changes, save the file and click 'Reload Config' to apply them.",
                parent=self.master
            )

        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Failed to open config file:\n{e}",
                parent=self.master
            )

    def _reload_config(self, parent_dialog=None):
        """Reload the config module to apply any changes."""
        try:
            # Reload the config module
            importlib.reload(config)

            # Re-import specific values that are used directly in this module
            # Note: We use config.VARIABLE_NAME elsewhere, so those will automatically update
            global DATA_DIR, OUT_DIR, TARGET_SERIES, NUM_COLUMNS, ALLOW_DUPLICATE_LABELS
            DATA_DIR = config.DATA_DIR
            OUT_DIR = config.OUT_DIR
            TARGET_SERIES = sanitize_target_series(config.TARGET_SERIES)
            NUM_COLUMNS = config.NUM_COLUMNS
            ALLOW_DUPLICATE_LABELS = config.ALLOW_DUPLICATE_LABELS

            # Update the UI elements if they need to reflect updated values
            # (input_dir and output_dir StringVars are set from these at startup)

            # Show success message with warning
            messagebox.showinfo(
                "Config Reloaded",
                "Configuration has been reloaded successfully.\n\n"
                "⚠️ Note: Some settings (loaded models, running processes, and certain UI elements) "
                "may require an application restart to fully take effect.",
                parent=parent_dialog if parent_dialog else self.master
            )

            # If parent dialog exists, refresh its content
            if parent_dialog:
                parent_dialog.destroy()
                self._view_configuration_dialog()

        except Exception as e:
            messagebox.showerror(
                "Reload Failed",
                f"Failed to reload configuration:\n\n{e}\n\n"
                "Please check the config file for syntax errors.",
                parent=parent_dialog if parent_dialog else self.master
            )

    def start_verification(self):
        """Start the verification workflow to review already classified studies."""
        output_dir = self.output_dir.get()
        if not os.path.isdir(output_dir):
            logger.error(f"Invalid output directory for verification: {output_dir}")
            messagebox.showerror("Error", "Please specify a valid output directory.", parent=self.master)
            return

        logger.info("Starting verification workflow")
        logger.info(f"Output directory: {output_dir}")

        self.is_running = True
        self.stop_requested.clear()
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.train_button.config(state="disabled")

        self.progress_frame = tk.Frame(self.master)
        self.progress_frame.pack(padx=20, pady=10, fill='x')

        self.progress_label = tk.StringVar(value="Starting verification...")
        tk.Label(self.progress_frame, textvariable=self.progress_label).pack(side="top")

        self.progress_bar = ttk.Progressbar(self.progress_frame, orient="horizontal", mode="determinate")
        self.progress_bar.pack(fill='x', pady=5)

        verification_thread = threading.Thread(
            target=self.run_verification_main,
            args=(output_dir,),
            daemon=True
        )
        verification_thread.start()
        self.master.after(100, self._poll_classification_status, verification_thread)

    def run_verification_main(self, output_dir):
        """Main verification workflow - aggregates classification results and displays for review."""
        try:
            import pandas as pd

            # Set up file logging now that we have the output directory
            os.makedirs(output_dir, exist_ok=True)
            update_log_file_location(output_dir)

            # 1. Aggregate all classification-results.csv files
            all_classifications = []
            study_folders = []

            for folder in os.listdir(output_dir):
                folder_path = os.path.join(output_dir, folder)
                if os.path.isdir(folder_path):
                    study_id = extract_study_id_from_path(folder_path)
                    csv_path = os.path.join(folder_path, get_study_csv_name(study_id))
                    if os.path.exists(csv_path):
                        try:
                            df = pd.read_csv(csv_path)
                            df['study_folder'] = folder_path
                            df['study_id'] = study_id
                            all_classifications.append(df)
                            study_folders.append(folder_path)
                        except Exception as e:
                            logger.warning(f"Error reading {csv_path}: {e}")

            if not all_classifications:
                self.master.after(0, lambda: messagebox.showinfo(
                    "No Classifications Found",
                    "No classified studies found in output directory.",
                    parent=self.master
                ))
                self.classification_finished(aborted=False)
                return

            # Combine all classifications
            all_df = pd.concat(all_classifications, ignore_index=True)

            # 2. Group by TARGET_SERIES
            series_groups = {}
            for target in TARGET_SERIES:
                if target == "Other":
                    continue
                target_df = all_df[all_df['Classification'] == target]
                if len(target_df) > 0:
                    series_groups[target] = target_df

            if not series_groups:
                self.master.after(0, lambda: messagebox.showinfo(
                    "No Series Found",
                    "No series found classified as any of the target series (excluding 'Other').",
                    parent=self.master
                ))
                self.classification_finished(aborted=False)
                return

            # 3. Calculate total verification windows
            max_per_window = NUM_COLUMNS * NUM_ROWS
            total_windows = sum((len(df) + max_per_window - 1) // max_per_window for df in series_groups.values())

            self.progress_bar.config(maximum=total_windows)
            current_window = 0

            # 4. Loop through each TARGET_SERIES
            studies_to_delete = set()

            for target_series in TARGET_SERIES:
                if target_series == "Other" or target_series not in series_groups:
                    continue

                if self.stop_requested.is_set():
                    self.classification_finished(aborted=True)
                    return

                target_df = series_groups[target_series]
                total_series = len(target_df)

                # Check if no series for this target
                if total_series == 0:
                    continue

                self.progress_label.set(f"Verifying {target_series} series ({total_series} total)...")

                # Split into batches
                for batch_start in range(0, total_series, max_per_window):
                    if self.stop_requested.is_set():
                        self.classification_finished(aborted=True)
                        return

                    batch_end = min(batch_start + max_per_window, total_series)
                    batch_df = target_df.iloc[batch_start:batch_end]

                    # Create verification window and wait for result
                    verification_done = threading.Event()
                    incorrect_studies = []
                    stop_all = [False]

                    # Show verification GUI on main thread
                    def show_verification():
                        show_verification_gui(
                            batch_df, target_series, batch_start, total_series,
                            self.stop_requested, verification_done, incorrect_studies, stop_all, output_dir
                        )

                    self.master.after(0, show_verification)

                    # Wait for user to complete verification
                    while not verification_done.is_set():
                        time.sleep(0.1)
                        if self.stop_requested.is_set():
                            self.classification_finished(aborted=True)
                            return

                    # Check if user pressed Stop All
                    if stop_all[0]:
                        self.stop_requested.set()
                        self.classification_finished(aborted=True)
                        return

                    # Add incorrect studies to deletion list
                    studies_to_delete.update(incorrect_studies)

                    # Update progress
                    current_window += 1
                    self.progress_bar.config(value=current_window)
                    self.master.update_idletasks()

            # 5. Handle deletion of incorrect studies
            if studies_to_delete:
                # Use a synchronous approach for confirmation dialog
                confirm_result = [None]
                confirm_event = threading.Event()

                def ask_confirm():
                    try:
                        response = messagebox.askyesno(
                            "Delete Incorrect Classifications",
                            f"You marked {len(studies_to_delete)} study folder(s) as incorrect.\n\n"
                            f"Are you sure you want to delete these entire study directories?\n\n"
                            f"This action cannot be undone.",
                            parent=self.master
                        )
                        confirm_result[0] = response
                    finally:
                        confirm_event.set()

                self.master.after(0, ask_confirm)
                confirm_event.wait()  # Wait for dialog to complete

                if confirm_result[0]:
                    deleted_count = 0
                    for study_folder in studies_to_delete:
                        try:
                            if os.path.exists(study_folder):
                                shutil.rmtree(study_folder)
                                deleted_count += 1
                                logger.debug(f"Deleted: {study_folder}")
                        except Exception as e:
                            logger.error(f"Error deleting {study_folder}: {e}")

                    self.master.after(0, lambda: messagebox.showinfo(
                        "Deletion Complete",
                        f"Successfully deleted {deleted_count} study folder(s).",
                        parent=self.master
                    ))

            self.classification_finished(aborted=False)

        except Exception as e:
            logger.exception(f"Error in verification workflow: {e}")
            self.classification_finished(aborted=True)


# At end of file, add asynchronous image loader and per-study classification GUI
from . import processing
from .processing import load_center_slice_fast


def _extract_dicom_path(path_entry):
    """
    Extract DICOM path from hierarchy entry.

    Args:
        path_entry: Either a string path or dict with 'dicom_path' key

    Returns:
        str: DICOM file path
    """
    if isinstance(path_entry, dict):
        return path_entry.get('dicom_path', '')
    return path_entry


def _extract_thumbnail_path(path_entry):
    """
    Extract thumbnail path from hierarchy entry if available.

    Args:
        path_entry: Either a string path or dict with 'thumbnail_path' key

    Returns:
        str or None: Thumbnail file path if available, None otherwise
    """
    if isinstance(path_entry, dict):
        return path_entry.get('thumbnail_path')
    return None


def _derive_thumbnail_path_from_dicom(dicom_path, output_dir):
    """
    Derive the thumbnail path from a DICOM file path.

    This function attempts to find the pre-generated thumbnail by parsing the DICOM path
    and looking for the corresponding image file in the series_thumbnails directory.
    Checks JPEG first, then PNG.

    Args:
        dicom_path: Path to DICOM file
        output_dir: Output directory containing series_thumbnails folder

    Returns:
        str or None: Path to thumbnail image if it exists, None otherwise
    """
    if not dicom_path or not output_dir:
        return None

    try:
        import pydicom

        # Read DICOM header to get series identifiers
        ds = pydicom.dcmread(dicom_path, stop_before_pixels=True, force=True)
        mrn = getattr(ds, "PatientID", None)
        study_uid = getattr(ds, "StudyInstanceUID", None)
        series_uid = getattr(ds, "SeriesInstanceUID", None)

        if not all([mrn, study_uid, series_uid]):
            return None

        # Construct thumbnail filename (without split index - most common case)
        thumbnail_dir = os.path.join(output_dir, "series_thumbnails")
        if not os.path.isdir(thumbnail_dir):
            return None

        # Try both thumbnail formats (JPEG first, then PNG)
        for extension in ['.jpg', '.png']:
            # Try without split index first
            thumbnail_filename = f"{mrn}_{study_uid}_{series_uid}{extension}"
            thumbnail_path = os.path.join(thumbnail_dir, thumbnail_filename)

            if os.path.exists(thumbnail_path):
                return thumbnail_path

            # Try with split indices (00-99)
            for split_idx in range(100):
                thumbnail_filename = f"{mrn}_{study_uid}_{series_uid}_{split_idx:02d}{extension}"
                thumbnail_path = os.path.join(thumbnail_dir, thumbnail_filename)
                if os.path.exists(thumbnail_path):
                    return thumbnail_path

        return None

    except Exception as e:
        # Silent failure - will fall back to runtime extraction
        return None


def show_verification_gui(batch_df, target_series, batch_start, total_series,
                         stop_event, done_event, incorrect_studies, stop_all, output_dir):
    """
    Display a verification GUI window for a batch of series.
    Shows middle slice thumbnails with Correct/Incorrect toggle buttons.

    Args:
        batch_df: DataFrame containing series to verify
        target_series: Name of the target series being verified
        batch_start: Starting index in the full series list
        total_series: Total number of series for this target
        stop_event: Event to signal stop requested
        done_event: Event to signal this window is complete
        incorrect_studies: List to append study folders marked as incorrect
        stop_all: List with single boolean element to signal Stop All
        output_dir: Output directory containing series_thumbnails folder
    """
    root = tk.Toplevel()
    enable_tkinter_autoscaling(root)

    batch_end = batch_start + len(batch_df)
    root.title(f"Verify {target_series} - Showing {batch_start+1}-{batch_end} of {total_series}")
    root.protocol("WM_DELETE_WINDOW", lambda: None)
    root.grab_set()

    # Main content frame
    content_frame = tk.Frame(root)
    content_frame.grid(row=0, column=0, padx=scale_value(20), pady=scale_value(20), sticky="nsew")
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)

    # Configure grid columns
    for i in range(NUM_COLUMNS):
        content_frame.grid_columnconfigure(i, weight=1)

    # Create placeholder image

    # Create loading indicator image (with text)
    loading_img = Image.new("RGB", (scale_value(256), scale_value(256)), (220, 220, 220))
    tk_loading = ImageTk.PhotoImage(loading_img)

    # Store toggle states and widgets
    toggle_states = {}  # path -> BooleanVar (True = Incorrect, False = Correct)
    toggle_buttons = {}  # path -> Button widget
    panel_widgets = {}  # path -> Label widget

    row_num = 0
    col_num = 0

    # Create grid of series thumbnails
    for idx, row_data in batch_df.iterrows():
        series_path = row_data['Path']
        series_desc = row_data.get('SeriesDescription', 'Unknown')
        study_folder = row_data['study_folder']

        # Create frame for this series
        series_frame = tk.Frame(content_frame, padx=scale_value(5), pady=scale_value(5))
        series_frame.grid(row=row_num, column=col_num, sticky="nsew")
        series_frame.grid_columnconfigure(0, weight=1)

        col_num += 1
        if col_num >= NUM_COLUMNS:
            col_num = 0
            row_num += 1

        # Image panel with loading indicator
        panel = tk.Label(series_frame, image=tk_loading, width=scale_value(256), height=scale_value(256))
        panel.image = tk_loading
        panel.grid(row=0, column=0, pady=(0, scale_value(5)))
        panel_widgets[series_path] = panel

        # Series description label
        tk.Label(series_frame, text=series_desc, wraplength=scale_value(250), justify="center").grid(
            row=1, column=0, pady=scale_value(5))

        # Toggle button for Correct/Incorrect
        toggle_var = tk.BooleanVar(value=False)  # False = Correct, True = Incorrect
        toggle_states[series_path] = (toggle_var, study_folder)

        def make_toggle_callback(var, btn, study_path, study_dir):
            def toggle():
                new_state = not var.get()
                var.set(new_state)
                if new_state:  # Incorrect
                    btn.config(text="Incorrect", bg="#ff4f4f", activebackground="#ff4f4f")
                else:  # Correct
                    btn.config(text="Correct", bg="#f0f0f0", activebackground="#f0f0f0")
            return toggle

        toggle_btn = tk.Button(
            series_frame,
            text="Correct",
            width=scale_value(15),
            bg="#f0f0f0",
            activebackground="#f0f0f0",
            command=None  # Will set below
        )
        toggle_btn.config(command=make_toggle_callback(toggle_var, toggle_btn, series_path, study_folder))
        toggle_btn.grid(row=2, column=0, pady=(0, scale_value(5)))
        toggle_buttons[series_path] = toggle_btn

    # Load all images in parallel using ThreadPoolExecutor
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def load_single_thumbnail(series_path, panel, output_dir):
        """Load a single thumbnail - optimized version with pre-generated thumbnail support (JPEG/PNG)."""
        try:
            pil_img = None

            # Try to load pre-generated thumbnail first
            thumbnail_path = _derive_thumbnail_path_from_dicom(series_path, output_dir)
            if thumbnail_path and os.path.exists(thumbnail_path):
                try:
                    from PIL import Image
                    pil_img = Image.open(thumbnail_path)
                except Exception as e:
                    logger.debug(f"Failed to load pre-generated thumbnail {thumbnail_path}: {e}")

            # Fall back to runtime extraction if thumbnail not available
            if pil_img is None:
                # Use the fast thumbnail loader
                pil_img = load_center_slice_fast(series_path)

            if pil_img is None:
                return None

            # Resize to target size
            target_size = scale_value(256)
            pil_img = pil_img.resize((target_size, target_size))

            return pil_img
        except Exception as e:
            logger.debug(f"Error loading thumbnail for {series_path}: {e}")
            return None

    def update_thumbnail(series_path, pil_img):
        """Update a panel with loaded image on main thread."""
        try:
            panel = panel_widgets.get(series_path)
            if panel is None or not panel.winfo_exists():
                return

            if pil_img is not None:
                pil_rgb = pil_img.convert('RGB') if hasattr(pil_img, 'convert') else pil_img
                tk_img = ImageTk.PhotoImage(pil_rgb, master=panel)
                panel.config(image=tk_img)
                panel.image = tk_img
        except Exception as e:
            logger.debug(f"Error updating thumbnail: {e}")

    # Start parallel loading in background
    def load_all_thumbnails():
        """Load all thumbnails in parallel."""
        with ThreadPoolExecutor(max_workers=NUM_PARALLEL_WORKERS) as executor:
            # Submit all loading tasks
            future_to_path = {
                executor.submit(load_single_thumbnail, row_data['Path'], panel_widgets[row_data['Path']], output_dir): row_data['Path']
                for idx, row_data in batch_df.iterrows()
            }

            # Update as each completes
            for future in as_completed(future_to_path):
                series_path = future_to_path[future]
                try:
                    pil_img = future.result()
                    if pil_img is not None:
                        # Schedule update on main thread
                        root.after(0, lambda p=series_path, img=pil_img: update_thumbnail(p, img))
                except Exception as e:
                    logger.debug(f"Error in parallel loading for {series_path}: {e}")

    # Start loading thread
    threading.Thread(target=load_all_thumbnails, daemon=True).start()

    # Button frame at bottom
    button_frame = tk.Frame(root, pady=scale_value(10))
    button_frame.grid(row=1, column=0, sticky="ew")
    button_frame.grid_columnconfigure(0, weight=1)
    button_frame.grid_columnconfigure(1, weight=0)
    button_frame.grid_columnconfigure(2, weight=0)
    button_frame.grid_columnconfigure(3, weight=0)
    button_frame.grid_columnconfigure(4, weight=1)

    def on_stop_all():
        stop_all[0] = True
        stop_event.set()
        root.grab_release()
        root.destroy()
        done_event.set()

    def on_skip():
        # Skip this batch without marking anything
        root.grab_release()
        root.destroy()
        done_event.set()

    def on_approve():
        # Collect all studies marked as incorrect
        marked_incorrect = set()
        for series_path, (toggle_var, study_folder) in toggle_states.items():
            if toggle_var.get():  # True = Incorrect
                marked_incorrect.add(study_folder)

        if marked_incorrect:
            # Confirm deletion
            response = messagebox.askyesno(
                "Mark for Deletion",
                f"{len(marked_incorrect)} series marked as incorrect.\n\n"
                f"Delete the entire output directories for these studies?\n\nYou will be asked to confirm at the end of verification.",
                parent=root
            )
            if response:
                incorrect_studies.extend(marked_incorrect)
            else:
                return

        root.grab_release()
        root.destroy()
        done_event.set()

    # Create buttons
    btn_stop = tk.Button(
        button_frame,
        bg="#e0a2a2",
        activebackground="#e0a2a2",
        text="Stop All (a)",
        font=("Arial", scale_value(14)),
        command=on_stop_all
    )
    btn_skip = tk.Button(
        button_frame,
        bg="#fff7a8",
        activebackground="#fff7a8",
        text="Skip (s)",
        font=("Arial", scale_value(14)),
        command=on_skip
    )
    btn_approve = tk.Button(
        button_frame,
        bg="#a2e0b2",
        activebackground="#a2e0b2",
        text="Approve (d)",
        font=("Arial", scale_value(14)),
        command=on_approve
    )

    # Grid buttons
    btn_stop.grid(row=0, column=1, padx=scale_value(10))
    btn_skip.grid(row=0, column=2, padx=scale_value(10))
    btn_approve.grid(row=0, column=3, padx=scale_value(10))

    # add hotkeys
    root.bind("<a>", lambda event: btn_stop.invoke())
    root.bind("<s>", lambda event: btn_skip.invoke())
    root.bind("<d>", lambda event: btn_approve.invoke())

    root.update()


def load_image_worker(dicom_paths, panel_label_ref, root):
    """Load a center slice (PIL Image) in a worker thread and schedule creation
    of the PhotoImage on the main thread. The worker only does PIL work; the
    PhotoImage is constructed on the Tk main thread to avoid invisible images.
    """
    # Calculate target size based on scale factor
    target_size = scale_value(256)

    try:
        pil_img = processing.load_center_slice(dicom_paths)
    except Exception:
        pil_img = None

    if pil_img is None:
        pil_img = Image.new('L', (target_size, target_size), 128)

    try:
        pil_img = pil_img.resize((target_size, target_size))
    except Exception:
        pass

    def update_panel():
        try:
            try:
                pil_rgb = pil_img.convert('RGB') if hasattr(pil_img, 'convert') else pil_img
            except Exception:
                pil_rgb = pil_img

            tk_img = ImageTk.PhotoImage(pil_rgb, master=panel_label_ref)
            panel_label_ref.config(image=tk_img)
            panel_label_ref.image = tk_img
        except Exception:
            # If anything goes wrong, leave the existing image in place
            try:
                _ = panel_label_ref.image
            except Exception:
                pass

    try:
        panel_label_ref.after(0, update_panel)
    except Exception:
        update_panel()


def classify_study_gui(study_dict, mrn, study_uid, output_root, stop_event, study_done_event, mode="Manual",
                       model_type=None, rf_model=None, rf_target_encoder=None, rf_feature_encoder=None, cnn_model=None,
                       cnn_target_encoder=None, rf_field_types=None, convert=True, accn=""):
    """
    Per-study classification GUI. Creates a grid of series thumbnails and dropdowns.
    Signals completion via study_done_event.
    """
    if stop_event.is_set():
        return

    root = tk.Toplevel()
    enable_tkinter_autoscaling(root)
    root.title(f"Classify Study ({mode}): MRN {mrn} | Accn {accn}")

    # Classification window appears fixed offset from main window
    root.update_idletasks()

    try:
        # Get main window position
        main_x = root.master.winfo_x()
        main_y = root.master.winfo_y()

        # Offset from main window
        x = main_x + scale_value(50)
        y = main_y + scale_value(50)

        root.geometry(f"+{x}+{y}")
    except Exception:
        # Fallback to screen center
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        window_width = root.winfo_reqwidth()
        window_height = root.winfo_reqheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        root.geometry(f"+{x}+{y}")

    root.protocol("WM_DELETE_WINDOW", lambda: None)
    root.grab_set()

    content_frame = tk.Frame(root)
    content_frame.grid(row=0, column=0, padx=scale_value(20), pady=scale_value(20), sticky="nsew")
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)

    for i in range(NUM_COLUMNS):
        content_frame.grid_columnconfigure(i, weight=1)

    dropdown_values = TARGET_SERIES
    dropdown_vars = {}
    panel_widgets = {}  # Store panel widgets for parallel loading

    # Precompute color map for the dropdown labels
    try:
        color_map = _generate_color_map(dropdown_values)
    except Exception:
        color_map = {k: (None, None) for k in dropdown_values}

    placeholder_img = Image.new("RGB", (scale_value(256), scale_value(256)), (200, 200, 200))
    tk_placeholder = ImageTk.PhotoImage(placeholder_img)

    optimized_assignments = {}
    predictions_by_path = {}
    if mode == "Semi-Auto":
        optimized_result = processing.optimize_study_selections(
            study_dict,
            model_type,
            rf_model, rf_target_encoder, rf_feature_encoder,
            cnn_model, cnn_target_encoder,
            rf_field_types  # Pass field types for consistent preprocessing
        )
        # optimize_study_selections returns (final_assignments, predictions_by_path)
        try:
            optimized_assignments, predictions_by_path = optimized_result
        except Exception:
            # Handle unexpected return format
            optimized_assignments = optimized_result if optimized_result else {}
            predictions_by_path = {}
        if not optimized_assignments:
            logger.warning("Optimization failed, falling back to 'Other'.")

    row_num = 0
    col_num = 0

    for series_desc, dicom_paths in study_dict.items():
        series_frame = tk.Frame(content_frame, padx=scale_value(5), pady=scale_value(5))
        series_frame.grid(row=row_num, column=col_num, sticky="nsew")
        series_frame.grid_columnconfigure(0, weight=1)

        col_num += 1
        if col_num >= NUM_COLUMNS:
            col_num = 0
            row_num += 1

        panel = tk.Label(series_frame, image=tk_placeholder, width=scale_value(256), height=scale_value(256))
        panel.image = tk_placeholder
        panel.grid(row=0, column=0, pady=(0, scale_value(5)))
        tk.Label(series_frame, text=series_desc, wraplength=scale_value(250), justify="center").grid(row=1, column=0, pady=scale_value(5))

        # Extract DICOM and thumbnail paths from hierarchy entry
        first_entry = dicom_paths[0]
        first_path = _extract_dicom_path(first_entry)
        thumbnail_path = _extract_thumbnail_path(first_entry)
        initial_value = "Other"
        if mode == "Semi-Auto" and first_path in optimized_assignments:
            initial_value = optimized_assignments[first_path]
            # Try to include probability info if available
            prob_info = predictions_by_path.get(first_path, {})
            combined_p = None
            rf_p = None
            cnn_p = None
            top_label = None
            top_p = None
            try:
                combined_scores = prob_info.get('combined', {}) if isinstance(prob_info, dict) else {}
                # find the top combined label/prob if present
                if isinstance(combined_scores, dict) and combined_scores:
                    try:
                        top_label, top_p = max(combined_scores.items(), key=lambda kv: kv[1])
                    except Exception:
                        top_label, top_p = None, None
                # prefer to show probabilities for the selected label when available
                if 'combined' in prob_info and initial_value in prob_info['combined']:
                    combined_p = prob_info['combined'][initial_value]
                if 'rf' in prob_info and initial_value in prob_info['rf']:
                    rf_p = prob_info['rf'][initial_value]
                if 'cnn' in prob_info and initial_value in prob_info['cnn']:
                    cnn_p = prob_info['cnn'][initial_value]
            except Exception:
                pass

            prob_parts = []
            if initial_value != 'Other':
                if combined_p is not None:
                    prob_parts.append(f"combined={combined_p:.2f}")
                if rf_p is not None:
                    prob_parts.append(f"rf={rf_p:.2f}")
                if cnn_p is not None:
                    prob_parts.append(f"cnn={cnn_p:.2f}")
            else:
                # If pre-selected as 'Other', show the top combined prediction (label + prob) if available
                if top_label is not None and top_p is not None:
                    prob_parts.append(f"best={top_label}, combined={top_p:.2f}")

            prob_str = (" (" + ", ".join(prob_parts) + ")") if prob_parts else ""
            logger.info(f"  -> Series '{series_desc}' semi-auto pre-selected as: {initial_value}{prob_str}")
        elif mode == "Manual":
            initial_value = "Other"

        variable = tk.StringVar(value=initial_value)

        # determine sensible defaults
        try:
            default_bg = series_frame.cget('bg') or '#f0f0f0'
        except Exception:
            default_bg = '#f0f0f0'
        default_fg = '#000000'

        # per-series colored dropdown implementation
        class ColoredDropdown:
            def __init__(self, parent, var, options, cmap, key=None, all_vars=None, mode=None):
                self.var = var
                self.options = list(options)
                self.cmap = cmap
                # identifier for this dropdown (first_path) and reference to the dict of all dropdown vars
                self.key = key
                self.all_vars = all_vars
                self.mode = mode
                self.btn = tk.Button(parent, text=self.var.get(), width=scale_value(20), relief='raised')
                self.menu = tk.Menu(parent, tearoff=0)

                for opt in self.options:
                    def _mk(v=opt):
                        def _cmd():
                            try:
                                self.var.set(v)
                            except Exception:
                                pass
                            # update this button's display
                            self._update_display(v)
                            # enforce uniqueness in Semi-Auto mode: if user selected a non-'Other' value,
                            # find any other dropdowns with that same value and set them to 'Other'
                            try:
                                self._handle_duplicates(v)
                            except Exception:
                                pass
                        return _cmd

                    cmd = _mk(opt)
                    if opt == 'Other':
                        try:
                            self.menu.add_command(label=opt, command=cmd)
                        except Exception:
                            pass
                    else:
                        try:
                            bg, fg = self.cmap.get(opt, (None, None))
                            if bg is not None:
                                self.menu.add_command(label=opt, command=cmd, background=bg, foreground=fg, activebackground=bg, activeforeground=fg)
                            else:
                                self.menu.add_command(label=opt, command=cmd)
                        except Exception:
                            try:
                                self.menu.add_command(label=opt, command=cmd)
                            except Exception:
                                pass

                def _on_click(event=None, menu=self.menu, btn=self.btn):
                    try:
                        menu.tk_popup(btn.winfo_rootx(), btn.winfo_rooty() + btn.winfo_height())
                    except Exception:
                        try:
                            menu.post(btn.winfo_rootx(), btn.winfo_rooty() + btn.winfo_height())
                        except Exception:
                            pass

                self.btn.configure(command=_on_click)

                try:
                    if hasattr(self.var, 'trace_add'):
                        self.var.trace_add('write', lambda *a: self._update_display())
                    else:
                        self.var.trace('w', lambda *a: self._update_display())
                except Exception:
                    pass

            def _handle_duplicates(self, val):
                # Enforce uniqueness based on AUTO_CLEAR_DUPLICATE_LABELS config
                # When enabled, selecting a non-'Other' label clears that label from all other dropdowns
                if ALLOW_DUPLICATE_LABELS or not self.all_vars or val == 'Other':
                    return
                try:
                    for k, other_var in list(self.all_vars.items()):
                        # skip our own entry
                        if k == self.key:
                            continue
                        try:
                            if other_var.get() == val:
                                other_var.set('Other')
                        except Exception:
                            pass
                except Exception:
                    pass

            def _update_display(self, v=None):
                val = self.var.get() if v is None else v
                if val == 'Other':
                    try:
                        self.btn.config(bg=default_bg, fg=default_fg, activebackground=default_bg, activeforeground=default_fg, text=val)
                    except Exception:
                        try:
                            self.btn.config(text=val)
                        except Exception:
                            pass
                    try:
                        self.btn.update_idletasks()
                    except Exception:
                        pass
                    return

                bg, fg = self.cmap.get(val, (None, None))
                try:
                    if bg is not None:
                        self.btn.config(bg=bg, fg=fg, activebackground=bg, activeforeground=fg, text=val)
                    else:
                        self.btn.config(bg=default_bg, fg=default_fg, text=val)
                    try:
                        self.btn.update_idletasks()
                    except Exception:
                        pass
                except Exception:
                    pass

        # create control row and widget
        control_row = tk.Frame(series_frame)
        cd = ColoredDropdown(control_row, variable, dropdown_values, color_map, key=first_path, all_vars=dropdown_vars, mode=mode)
        try:
            cd.btn.pack(side='left')
        except Exception:
            try:
                cd.btn.grid(row=0, column=0)
            except Exception:
                pass

        try:
            control_row.grid(row=2, column=0, pady=(0, scale_value(5)))
        except Exception:
            try:
                control_row.pack(pady=(0, scale_value(5)))
            except Exception:
                pass

        try:
            cd._update_display()
        except Exception:
            pass

        dropdown_vars[first_path] = variable
        panel_widgets[first_path] = (panel, dicom_paths, thumbnail_path)  # Store for parallel loading

        # Enforce uniqueness for initial assignments based on ALLOW_DUPLICATE_LABELS config
        # If enabled and this dropdown's initial value duplicates an existing assignment, reset it to 'Other'
        try:
            if not ALLOW_DUPLICATE_LABELS and variable.get() != 'Other':
                for k, ov in dropdown_vars.items():
                    if k == first_path:
                        continue
                    try:
                        if ov.get() == variable.get():
                            variable.set('Other')
                            try:
                                cd._update_display()
                            except Exception:
                                pass
                            break
                    except Exception:
                        pass
        except Exception:
            pass

    # Load all images in parallel using ThreadPoolExecutor (same as verification mode)
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def load_single_thumbnail(first_path, panel, dicom_paths, thumbnail_path):
        """Load a single thumbnail - optimized version with pre-generated thumbnail support (JPEG/PNG)."""
        try:
            pil_img = None

            # Try to load pre-generated thumbnail first
            if thumbnail_path and os.path.exists(thumbnail_path):
                try:
                    from PIL import Image
                    pil_img = Image.open(thumbnail_path)
                except Exception as e:
                    logger.debug(f"Failed to load pre-generated thumbnail {thumbnail_path}: {e}")

            # Fall back to runtime extraction if thumbnail not available
            if pil_img is None:
                # Extract DICOM paths from hierarchy entries
                dicom_path_list = []
                for entry in dicom_paths:
                    dicom_path_list.append(_extract_dicom_path(entry))

                # Use the fast thumbnail loader
                pil_img = load_center_slice_fast(first_path)

                if pil_img is None:
                    # Fallback to original method
                    pil_img = processing.load_center_slice(dicom_path_list)

            if pil_img is None:
                return None

            # Resize to target size
            target_size = scale_value(256)
            pil_img = pil_img.resize((target_size, target_size))

            return pil_img
        except Exception as e:
            logger.debug(f"Error loading thumbnail for {first_path}: {e}")
            return None

    def update_thumbnail(first_path, pil_img):
        """Update a panel with loaded image on main thread."""
        try:
            panel_data = panel_widgets.get(first_path)
            if panel_data is None:
                return

            panel = panel_data[0]
            if not panel.winfo_exists():
                return

            if pil_img is not None:
                pil_rgb = pil_img.convert('RGB') if hasattr(pil_img, 'convert') else pil_img
                tk_img = ImageTk.PhotoImage(pil_rgb, master=panel)
                panel.config(image=tk_img)
                panel.image = tk_img
        except Exception as e:
            logger.debug(f"Error updating thumbnail: {e}")

    # Start parallel loading in background
    def load_all_thumbnails():
        """Load all thumbnails in parallel."""
        with ThreadPoolExecutor(max_workers=NUM_PARALLEL_WORKERS) as executor:
            # Submit all loading tasks
            future_to_path = {
                executor.submit(load_single_thumbnail, path, panel_data[0], panel_data[1], panel_data[2]): path
                for path, panel_data in panel_widgets.items()
            }

            # Update as each completes
            for future in as_completed(future_to_path):
                series_path = future_to_path[future]
                try:
                    pil_img = future.result()
                    if pil_img is not None:
                        # Schedule update on main thread
                        root.after(0, lambda p=series_path, img=pil_img: update_thumbnail(p, img))
                except Exception as e:
                    logger.debug(f"Error in parallel loading for {series_path}: {e}")

    # Start loading thread
    threading.Thread(target=load_all_thumbnails, daemon=True).start()

    button_frame = tk.Frame(root, pady=scale_value(10))
    button_frame.grid(row=1, column=0, sticky="ew")
    button_frame.grid_columnconfigure(0, weight=1)
    button_frame.grid_columnconfigure(1, weight=0)
    button_frame.grid_columnconfigure(2, weight=0)
    button_frame.grid_columnconfigure(3, weight=0)
    button_frame.grid_columnconfigure(4, weight=1)

    def on_done():
        worker_args = (mrn, study_uid, output_root, dropdown_vars, study_dict, accn)

        conversion_thread = threading.Thread(
            target=processing.background_conversion_worker,
            args=worker_args + (convert,),
            daemon=True
        )
        conversion_thread.start()

        root.grab_release()
        root.destroy()
        study_done_event.set()

    def on_stop():
        stop_event.set()
        root.grab_release()
        root.destroy()
        study_done_event.set()

    def on_skip():
        # Skip this study without saving or conversion; just advance to next study
        try:
            root.grab_release()
        except Exception:
            pass
        try:
            root.destroy()
        except Exception:
            pass
        try:
            study_done_event.set()
        except Exception:
            pass

    # make buttons
    btn_stop = tk.Button(button_frame, bg="#e0a2a2", activebackground="#e0a2a2", text="Stop All (a)", font=("Arial", scale_value(14)), command=on_stop)
    btn_skip = tk.Button(button_frame, bg="#fff7a8", activebackground="#fff7a8", text="Skip (s)", font=("Arial", scale_value(14)), command=on_skip)
    btn_done = tk.Button(button_frame, bg="#a2e0b2", activebackground="#a2e0b2", text="Approve (d)", font=("Arial", scale_value(14)), command=on_done)

    # grid buttons
    btn_stop.grid(row=0, column=1, padx=scale_value(10))
    btn_skip.grid(row=0, column=2, padx=scale_value(10))
    btn_done.grid(row=0, column=3, padx=scale_value(10))

    # add hotkeys
    root.bind("<a>", lambda event: btn_stop.invoke())
    root.bind("<s>", lambda event: btn_skip.invoke())
    root.bind("<d>", lambda event: btn_done.invoke())

    root.update()
