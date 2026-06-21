import os
import json
import threading
import time
import random
import tkinter as tk
from tkinter import filedialog
import customtkinter as ctk
from . import themed_dialogs as messagebox
from . import config
from . import ui_theme
from .ui_theme import (scale_value, scale_thumbnail, cascade, place_main_window,
                       ThemedProgressBar, TitledFrame)
from .series_scroller import attach_series_scrolling
from .config import NUM_COLUMNS, NUM_ROWS, ALLOW_DUPLICATE_LABELS, THUMBNAIL_SCALE_FACTOR
from .utils import (extract_study_id_from_path, get_study_csv_name, get_study_dir_name,
                     get_nifti_name, get_nifti_prefix, sanitize_target_series,
                     get_classified_outputs_dir, get_num_parallel_workers)
from .logging_config import get_logger, setup_logging, update_log_file_location
from PIL import Image, ImageTk
import shutil
import colorsys
import platform
import subprocess
import importlib

logger = get_logger()


class _LazyModule:
    """Import a module on first attribute access.

    Heavy dependencies (torch, scikit-learn, pandas, pydicom, joblib) are only
    needed once the user trains/loads a model or scans DICOMs. Deferring them
    keeps `import mrid_app.gui` — and therefore the time before the main window
    appears — fast. Call sites use these exactly like the real module."""

    def __init__(self, name, package=None):
        self.__dict__['_name'] = name
        self.__dict__['_package'] = package
        self.__dict__['_mod'] = None

    def _load(self):
        if self._mod is None:
            self.__dict__['_mod'] = importlib.import_module(self._name, self._package)
        return self._mod

    def __getattr__(self, attr):
        return getattr(self._load(), attr)


# Lazily-imported modules (see _LazyModule). pydicom/joblib are stdlib-light
# but still cost ~250 ms combined at import; processing pulls in torch+sklearn.
processing = _LazyModule('.processing', __package__)
pydicom = _LazyModule('pydicom')
joblib = _LazyModule('joblib')


def load_center_slice_fast(*args, **kwargs):
    """Thin wrapper so this name resolves without importing processing eagerly."""
    return processing.load_center_slice_fast(*args, **kwargs)


# Sanitize TARGET_SERIES to ensure safe filenames
TARGET_SERIES = sanitize_target_series(config.TARGET_SERIES)


# ---------------------------------------------------------------------------
# Persistent last-used directory helpers
# ---------------------------------------------------------------------------
# Project root is one level above this file (mrid_app/gui.py → mrid/)
_PREFS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".mrid_prefs.json")


def _load_last_dirs():
    """Return (input_dir, output_dir) from the prefs file, falling back to home."""
    home = os.path.expanduser("~")
    try:
        if os.path.isfile(_PREFS_FILE):
            with open(_PREFS_FILE, "r") as fh:
                prefs = json.load(fh)
            input_dir = prefs.get("last_input_dir", home)
            output_dir = prefs.get("last_output_dir", home)
            # Fall back to home if stored path no longer exists
            if not os.path.isdir(input_dir):
                input_dir = home
            if not os.path.isdir(output_dir):
                output_dir = home
            return input_dir, output_dir
    except Exception:
        pass
    return home, home


def _save_last_dirs(input_dir: str, output_dir: str):
    """Persist input_dir and output_dir to the prefs file."""
    try:
        prefs = {}
        if os.path.isfile(_PREFS_FILE):
            try:
                with open(_PREFS_FILE, "r") as fh:
                    prefs = json.load(fh)
            except Exception:
                prefs = {}
        prefs["last_input_dir"] = input_dir
        prefs["last_output_dir"] = output_dir
        with open(_PREFS_FILE, "w") as fh:
            json.dump(prefs, fh, indent=2)
    except Exception as exc:
        logger.debug(f"Could not save last-used directories to {_PREFS_FILE}: {exc}")


class _ToolTip:
    """Simple hover tooltip for a tkinter widget."""

    def __init__(self, widget, text=""):
        self.widget = widget
        self.text = text
        self._tip_win = None
        widget.bind("<Enter>", self._show, add="+")
        widget.bind("<Leave>", self._hide, add="+")

    def update_text(self, text):
        self.text = text

    def _show(self, _event=None):
        if not self.text:
            return
        try:
            x = self.widget.winfo_rootx() + 20
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
            self._tip_win = tw = tk.Toplevel(self.widget)
            tw.wm_overrideredirect(True)
            tw.wm_geometry(f"+{x}+{y}")
            tk.Label(tw, text=self.text, justify="left",
                     background=ui_theme.TOOLTIP_BG, foreground=ui_theme.TOOLTIP_FG,
                     relief="solid", borderwidth=1,
                     font=("TkDefaultFont", 9)).pack()
        except Exception:
            pass

    def _hide(self, _event=None):
        try:
            if self._tip_win:
                self._tip_win.destroy()
                self._tip_win = None
        except Exception:
            pass


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
        self._cancelled = False
        self.root = ctk.CTkToplevel(master)
        self.root.title("Model Training In Progress...")
        cascade(self.root, master)
        self.root.protocol("WM_DELETE_WINDOW", self.do_nothing)
        self.root.grab_set()
        self.label = tk.StringVar(value="Initializing training process...")
        ctk.CTkLabel(self.root, textvariable=self.label).pack(pady=scale_value(10))
        self.progress = ThemedProgressBar(self.root, length=400, mode="indeterminate")
        self.progress.pack(pady=scale_value(5), padx=scale_value(20))
        self.progress.start(10)
        self.cancel_button = ctk.CTkButton(
            self.root,
            text="Cancel",
            command=self._on_cancel,
            fg_color=ui_theme.RED,
            hover_color=ui_theme.RED_HOVER,
            width=100,
        )
        self.cancel_button.pack(pady=scale_value(8))
        # Pin the size to the built content so status-text updates don't
        # resize the window (real pixels, so bypass CTk geometry scaling)
        self.root.update_idletasks()
        width = max(self.root.winfo_reqwidth(), scale_value(460))
        self.root.wm_geometry(f"{width}x{self.root.winfo_reqheight()}")
        self.root.update()

    def _on_cancel(self):
        """Called when the Cancel button is pressed."""
        self._cancelled = True
        self.cancel_button.configure(state="disabled", text=ui_theme.t("Cancelling…"))
        self.label.set("Cancelling – please wait…")

    def is_cancelled(self):
        """Return True if the user has requested cancellation."""
        return self._cancelled

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
            try:
                if hasattr(self, 'progress') and self.progress.winfo_exists():
                    try:
                        self.progress.stop()
                    except Exception:
                        pass
            except Exception:
                pass
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
            self.root = ctk.CTkToplevel(tk._default_root)
        except AttributeError:
            self.root = ctk.CTkToplevel()
        try:
            self.root.title(title)
        except Exception:
            pass
        # No fixed size: the window grows as the organize/thumbnail bars appear
        cascade(self.root, tk._default_root)
        self.root.protocol("WM_DELETE_WINDOW", self.do_nothing)

        # First progress bar: Scanning files
        self.label = ctk.CTkLabel(self.root, text=label_text)
        self.label.pack(pady=scale_value(5))
        self.progress = ThemedProgressBar(self.root, length=400, mode="determinate",
                                          maximum=total_files)
        self.progress.pack(pady=scale_value(3), padx=scale_value(30))
        self.count_label = ctk.CTkLabel(self.root, text="0 / 0")
        self.count_label.pack(pady=scale_value(3))

        # Second progress bar: Organizing series (initially hidden)
        self.organize_label = ctk.CTkLabel(self.root, text="")
        self.organize_progress = ThemedProgressBar(self.root, length=400, mode="determinate",
                                                   maximum=100)
        self.organize_count_label = ctk.CTkLabel(self.root, text="")

        # Third progress bar: Generating thumbnails (initially hidden)
        self.thumbnail_label = ctk.CTkLabel(self.root, text="")
        self.thumbnail_progress = ThemedProgressBar(self.root, length=400, mode="determinate",
                                                    maximum=100)
        self.thumbnail_count_label = ctk.CTkLabel(self.root, text="")

        self.root.update()

    def update(self, value, total=None):
        """Update first progress bar with counter text."""
        self.progress["value"] = value
        if total is None:
            total = self.progress["maximum"]
        self.count_label.configure(text=f"{value} / {int(total)}")
        self.root.update_idletasks()

    def show_organize_progress(self, label_text="Organizing series...", total=1):
        """Show the second progress bar for organizing series."""
        self.organize_label.configure(text=label_text)
        self.organize_label.pack(pady=scale_value(5))
        self.organize_progress["maximum"] = total
        self.organize_progress.pack(pady=scale_value(3), padx=scale_value(30))
        self.organize_count_label.pack(pady=scale_value(3))
        self.root.update_idletasks()

    def update_organize_progress(self, current, total):
        """Update organizing progress bar and count text."""
        self.organize_progress["maximum"] = total
        self.organize_progress["value"] = current
        self.organize_count_label.configure(text=f"{current} / {total}")
        self.root.update_idletasks()

    def show_thumbnail_progress(self, label_text="Generating thumbnails...", total=1):
        """Show the third progress bar for thumbnail generation."""
        self.thumbnail_label.configure(text=label_text)
        self.thumbnail_label.pack(pady=scale_value(5))
        self.thumbnail_progress["maximum"] = total
        self.thumbnail_progress.pack(pady=scale_value(3), padx=scale_value(30))
        self.thumbnail_count_label.pack(pady=scale_value(3))
        self.root.update_idletasks()

    def update_thumbnail_progress(self, current, total):
        """Update thumbnail progress bar and count text."""
        self.thumbnail_progress["maximum"] = total
        self.thumbnail_progress["value"] = current
        self.thumbnail_count_label.configure(text=f"{current} / {total}")
        self.root.update_idletasks()

    def close(self):
        self.root.destroy()

    def do_nothing(self):
        pass


class MainApp:
    def __init__(self, master):
        self.master = master
        master.title("MRID — DICOM Classification Tool")
        _default_input, _default_output = _load_last_dirs()
        self.input_dir = tk.StringVar(value=_default_input)
        self.output_dir = tk.StringVar(value=_default_output)
        self.classification_mode = tk.StringVar(value="Manual")
        self.model_type = tk.StringVar(value="Header + Image Model")
        self.rf_model = None
        self.rf_target_encoder = None
        self.rf_feature_encoder = None
        self.rf_field_types = None  # Store field types for consistent preprocessing
        self.cnn_model = None
        self.cnn_target_encoder = None
        self.stop_requested = threading.Event()
        self.is_running = False
        self.convert_to_nifti = tk.BooleanVar(value=False)
        # Track the currently active config file path
        self.active_config_path = None  # None means using repository config
        # Analyze-mode training state
        self._analyze_csv_ready = False       # True once selfsupervised CSV is written
        self._selfsupervised_csv_path = None  # Path to the written CSV

        menubar = tk.Menu(master)
        ui_theme.style_menu(menubar)
        master.config(menu=menubar)

        other_menu = tk.Menu(menubar, tearoff=0)
        ui_theme.style_menu(other_menu)
        menubar.add_cascade(label="File Options", menu=other_menu)
        other_menu.add_command(label="Convert NIfTIs", command=self._convert_niftis_utility)
        other_menu.add_command(label="Export DICOMS", command=self._export_dicoms_utility)
        other_menu.add_command(label="Erase Data", command=self._erase_data_utility)

        config_menu = tk.Menu(menubar, tearoff=0)
        ui_theme.style_menu(config_menu)
        menubar.add_cascade(label="Configuration", menu=config_menu)
        config_menu.add_command(label="Run Config Wizard", command=self._run_config_wizard)
        config_menu.add_command(label="Save Config File", command=self._save_config)
        config_menu.add_command(label="Load Config File", command=self._load_config)
        config_menu.add_command(label="Edit Config File", command=self._edit_config_file)
        config_menu.add_command(label="Reload Config File", command=self._reload_config_menu)
        config_menu.add_command(label="Add Missing Parameters", command=self._add_missing_params_menu)

        # UI layout
        dir_frame = TitledFrame(master, title="Directories")
        dir_frame.pack(padx=scale_value(20), pady=scale_value(10), fill="x")
        dirs = dir_frame.content
        ctk.CTkLabel(dirs, text="Input DIR:").grid(row=0, column=0, padx=scale_value(5), pady=scale_value(5), sticky="w")
        ctk.CTkEntry(dirs, textvariable=self.input_dir, width=420).grid(row=0, column=1, padx=scale_value(5), pady=scale_value(5), sticky="ew")
        ctk.CTkButton(dirs, text="Browse", width=90, command=lambda: self.browse_dir(self.input_dir, "Select INPUT directory")).grid(row=0, column=2, padx=scale_value(5), pady=scale_value(5))
        ctk.CTkLabel(dirs, text="Output DIR:").grid(row=1, column=0, padx=scale_value(5), pady=scale_value(5), sticky="w")
        ctk.CTkEntry(dirs, textvariable=self.output_dir, width=420).grid(row=1, column=1, padx=scale_value(5), pady=scale_value(5), sticky="ew")
        ctk.CTkButton(dirs, text="Browse", width=90, command=lambda: self.browse_dir(self.output_dir, "Select OUTPUT directory")).grid(row=1, column=2, padx=scale_value(5), pady=scale_value(5))
        dirs.columnconfigure(1, weight=1)

        control_frame = ctk.CTkFrame(master, fg_color="transparent")
        control_frame.pack(fill="x", padx=scale_value(20), pady=scale_value(10))
        # Mode selector at the far left, before the Start button
        mode_frame = ctk.CTkFrame(control_frame, fg_color="transparent")
        mode_frame.pack(side="left", padx=scale_value(5))
        ctk.CTkLabel(mode_frame, text="Mode:").pack(side="left", padx=scale_value(5))
        mode_options = ["Analyze", "Manual", "Semi-Auto", "Full-Auto", "Verify"]
        mode_menu = ctk.CTkOptionMenu(mode_frame, variable=self.classification_mode,
                                      values=mode_options, width=130,
                                      command=self._check_model_status)
        mode_menu.pack(side="left", padx=scale_value(5))
        model_type_frame = ctk.CTkFrame(control_frame, fg_color="transparent")
        model_type_frame.pack(side="right", padx=scale_value(5))
        ctk.CTkLabel(model_type_frame, text="Model Type:").pack(side="left", padx=scale_value(5))
        model_options = ["DICOM Header Model", "Image-Based Model", "Header + Image Model"]
        model_menu = ctk.CTkOptionMenu(model_type_frame, variable=self.model_type,
                                       values=model_options, width=200,
                                       command=self._check_model_status)
        model_menu.pack(side="left", padx=scale_value(5))

        # Convert toggle (Yes/No)
        convert_frame = ctk.CTkFrame(control_frame, fg_color="transparent")
        convert_frame.pack(side="right", padx=scale_value(10))
        ctk.CTkLabel(convert_frame, text="Convert to NIfTI:").pack(side="left", padx=(0, scale_value(5)))

        # Single button toggle showing Yes/No. Clicking flips the underlying BooleanVar
        def _convert_button_toggle():
            new_state = not bool(self.convert_to_nifti.get())
            self.convert_to_nifti.set(new_state)
            # call the existing handler which performs dcm2niix check and may revert
            self._on_toggle_convert()
            # reflect final state on the button text
            try:
                self._convert_button.configure(text=("Yes" if self.convert_to_nifti.get() else "No"))
            except Exception:
                pass

        self._convert_button = ctk.CTkButton(convert_frame, width=60,
                                             fg_color=ui_theme.NEUTRAL, hover_color=ui_theme.NEUTRAL_HOVER,
                                             text=("Yes" if self.convert_to_nifti.get() else "No"),
                                             command=_convert_button_toggle)
        self._convert_button.pack(side="left")

        button_font = ctk.CTkFont(size=13, weight="bold")
        # Fixed width sized for the longest text ("▶ Start Classification") so
        # mode changes never resize the window
        self.start_button = ctk.CTkButton(control_frame, text=ui_theme.t("▶ Start Classification"), font=button_font,
                                          width=190, command=self.start_classification, state="normal")
        self.start_button.pack(side="left", padx=scale_value(5))
        self.stop_button = ctk.CTkButton(control_frame, text="Stop", width=70,
                                         fg_color=ui_theme.NEUTRAL, hover_color=ui_theme.NEUTRAL_HOVER,
                                         command=self.stop_classification, state="disabled")
        self.stop_button.pack(side="left", padx=scale_value(5))
        self.train_button = ctk.CTkButton(control_frame, text="Train Model(s)", width=120,
                                          command=self._train_model)
        self.train_button.pack(side="right", padx=scale_value(5))
        self._train_button_tooltip = _ToolTip(self.train_button, "")
        self.load_button = ctk.CTkButton(control_frame, text="Load Model(s)", width=120,
                                         command=self._load_model_handler)
        self.load_button.pack(side="right", padx=scale_value(5))
        self._load_button_tooltip = _ToolTip(self.load_button, "")
        self._check_model_status()
        # Size the window to fit the UI and place it (CTk's fixed default
        # geometry would otherwise clip the control row)
        place_main_window(master)

    def browse_dir(self, var, title):
        directory = filedialog.askdirectory(title=title, initialdir=var.get())
        if directory:
            # If the output directory is changing, unload any loaded models so
            # they are not accidentally used against a different project/dataset.
            if var is self.output_dir and directory != var.get():
                self._unload_models()
            var.set(directory)

    def _unload_models(self):
        """Unload all classification models and update UI state."""
        self.rf_model = None
        self.rf_target_encoder = None
        self.rf_feature_encoder = None
        self.rf_field_types = None
        self.cnn_model = None
        self.cnn_target_encoder = None
        logger.info("Classification models unloaded (output directory changed).")
        self._check_model_status()

    def _on_toggle_convert(self):
        """Handler for the convert-to-nifti toggle. If enabling, check for dcm2niix in PATH.
        If not found, revert toggle and show an error dialog."""
        enabled = bool(self.convert_to_nifti.get())
        try:
            if hasattr(self, '_convert_button'):
                self._convert_button.configure(text=("Yes" if enabled else "No"))
        except Exception:
            pass

        if enabled:
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
                        self._convert_button.configure(text="No")
                except Exception:
                    pass
                return

    def _load_model_handler(self):
        """Handle Load Model(s) button press. If models don't exist, ask user if they want to train them."""
        output_dir = self.output_dir.get()
        model_type = self.model_type.get()
        rf_needed = model_type in ["DICOM Header Model", "Header + Image Model"]
        cnn_needed = model_type in ["Image-Based Model", "Header + Image Model"]

        model_files_dir = os.path.join(output_dir, "model_files")
        rf_model_path = os.path.join(model_files_dir, "rf_classifier_model.joblib")
        cnn_model_path = os.path.join(model_files_dir, "cnn_classifier_model.pth")

        rf_exists = os.path.exists(rf_model_path) if rf_needed else True
        cnn_exists = os.path.exists(cnn_model_path) if cnn_needed else True

        if rf_exists and cnn_exists:
            if rf_needed:
                self._load_classifier_and_encoder()
            if cnn_needed:
                self._load_image_classifier()
            self._check_model_status()
            return

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

        # Models load automatically when training completes
        if rf_needed and not rf_exists:
            self.train_classifier_rf()
        if cnn_needed and not cnn_exists:
            self.train_classifier_cnn()

    def _load_classifier_and_encoder(self):
        output_dir = self.output_dir.get()
        model_files_dir = os.path.join(output_dir, "model_files")
        model_path = os.path.join(model_files_dir, "rf_classifier_model.joblib")
        target_encoder_path = os.path.join(model_files_dir, "label_encoder.joblib")
        feature_encoder_path = os.path.join(model_files_dir, "feature_encoder.joblib")
        field_types_path = os.path.join(model_files_dir, "field_types.joblib")

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
        model_files_dir = os.path.join(output_dir, "model_files")
        model_path = os.path.join(model_files_dir, "cnn_classifier_model.pth")
        target_encoder_path = os.path.join(model_files_dir, "label_encoder.joblib")
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

            # Best available device: CUDA (NVIDIA/AMD ROCm) > Apple MPS > CPU
            from .cnn import get_torch_device, load_cnn_model
            map_loc = get_torch_device()
            logger.info(f"Loading CNN model on device: {map_loc}")
            self.cnn_model = load_cnn_model(model_path, num_classes, map_location=map_loc)
            logger.info("Image-Based Model (CNN) loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load CNN model: {e}")
            messagebox.showerror("Load Error", f"Failed to load CNN model: {e}", parent=self.master)
            self.cnn_model = None
            self.cnn_target_encoder = None

    def _count_classified_cases(self):
        """Return the number of study folders in classified_outputs that contain a classification-results CSV."""
        output_dir = self.output_dir.get()
        if not output_dir:
            return 0
        classified_dir = get_classified_outputs_dir(output_dir)
        if not os.path.isdir(classified_dir):
            return 0
        count = 0
        try:
            for name in os.listdir(classified_dir):
                folder = os.path.join(classified_dir, name)
                if not os.path.isdir(folder):
                    continue
                csv_name = get_study_csv_name(name)
                if os.path.isfile(os.path.join(folder, csv_name)):
                    count += 1
        except Exception:
            pass
        return count

    def _check_model_status(self, event=None):
        model_type = self.model_type.get()
        mode = self.classification_mode.get()
        rf_loaded = self.rf_model is not None and self.rf_target_encoder is not None
        cnn_loaded = self.cnn_model is not None and self.cnn_target_encoder is not None
        can_start = False
        models_loaded = False
        mode_is_manual = False
        mode_is_verify = False
        mode_is_analyze = False

        if mode == "Analyze":
            can_start = True
            mode_is_analyze = True
        elif mode == "Manual":
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
            if mode_is_analyze:
                self.start_button.configure(text=ui_theme.t("▶ Start Analysis"), state="normal", fg_color=ui_theme.GREEN, hover_color=ui_theme.GREEN_HOVER)
            elif mode_is_verify:
                self.start_button.configure(text=ui_theme.t("▶ Start Verification"), state="normal", fg_color=ui_theme.GREEN, hover_color=ui_theme.GREEN_HOVER)
            else:
                self.start_button.configure(text=ui_theme.t("▶ Start Classification"), state="normal", fg_color=ui_theme.GREEN, hover_color=ui_theme.GREEN_HOVER)
        else:
            self.start_button.configure(text=ui_theme.t("▶ Start Classification"), state="disabled", fg_color=ui_theme.DISABLED)

        # Update load button
        _MIN_CASES = 10
        if models_loaded:
            self.load_button.configure(text="Model(s) Loaded", state="disabled", fg_color=ui_theme.GREEN)
            load_tip = ""
        elif mode_is_analyze:
            self.load_button.configure(text="Load Model(s)", state="disabled", fg_color=ui_theme.DISABLED)
            load_tip = "Cannot load models in Analyze mode"
        elif mode_is_verify:
            self.load_button.configure(text="Load Model(s)", state="disabled", fg_color=ui_theme.DISABLED)
            load_tip = "Cannot load models in Verify mode"
        elif mode_is_manual:
            self.load_button.configure(text="Load Model(s)", state="disabled", fg_color=ui_theme.DISABLED)
            load_tip = "Cannot load models in Manual mode"
        else:
            # Semi-Auto / Full-Auto: check whether model files exist
            output_dir = self.output_dir.get()
            model_files_dir = os.path.join(output_dir, "model_files")
            rf_needed = model_type in ["DICOM Header Model", "Header + Image Model"]
            cnn_needed = model_type in ["Image-Based Model", "Header + Image Model"]
            rf_exists = os.path.exists(os.path.join(model_files_dir, "rf_classifier_model.joblib")) if rf_needed else True
            cnn_exists = os.path.exists(os.path.join(model_files_dir, "cnn_classifier_model.pth")) if cnn_needed else True
            models_exist = rf_exists and cnn_exists
            n_cases = self._count_classified_cases()

            if models_exist:
                # Models on disk — always allow load
                self.load_button.configure(text="Load Model(s)", state="normal", fg_color=ui_theme.RED, hover_color=ui_theme.RED_HOVER)
                load_tip = ""
            elif n_cases >= _MIN_CASES:
                # No models yet but enough data to train — allow (will offer to train)
                self.load_button.configure(text="Load Model(s)", state="normal", fg_color=ui_theme.RED, hover_color=ui_theme.RED_HOVER)
                load_tip = ""
            else:
                # No models and not enough data yet
                self.load_button.configure(text="Load Model(s)", state="disabled", fg_color=ui_theme.DISABLED)
                load_tip = f"Classify at least {_MIN_CASES} cases to enable model loading/training"

        try:
            self._load_button_tooltip.update_text(load_tip)
        except Exception:
            pass

        # ── Train button state ────────────────────────────────────────────────
        _MIN_CASES = 10
        if mode == "Analyze":
            train_available = self._analyze_csv_ready
            tip = "" if train_available else "Analyze first"
        elif mode == "Verify":
            train_available = False
            tip = "Cannot train models in Verify mode"
        else:
            # Manual / Semi-Auto / Full-Auto: need ≥ _MIN_CASES classified cases
            n_cases = self._count_classified_cases()
            train_available = n_cases >= _MIN_CASES
            tip = "" if train_available else f"Classify at least {_MIN_CASES} cases to enable model training"

        try:
            self._train_button_tooltip.update_text(tip)
        except Exception:
            pass

        if train_available:
            self.train_button.configure(state="normal", fg_color=ui_theme.GREEN, hover_color=ui_theme.GREEN_HOVER)
        else:
            self.train_button.configure(state="disabled", fg_color=ui_theme.DISABLED)

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

        # In Analyze mode, train exclusively from the self-supervised CSV
        mode = self.classification_mode.get()
        if mode == "Analyze":
            if not self._analyze_csv_ready or not self._selfsupervised_csv_path:
                messagebox.showwarning("Not Ready", "Please run Analyze first to generate the self-supervised training data.", parent=self.master)
                return
            if rf_needed:
                self.train_classifier_rf(preloaded_csv=self._selfsupervised_csv_path)
            if cnn_needed:
                self.train_classifier_cnn(preloaded_csv=self._selfsupervised_csv_path)
            return

        if rf_needed:
            self.train_classifier_rf()
        if cnn_needed:
            self.train_classifier_cnn()

    def train_classifier_rf(self, preloaded_csv=None):
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

        # Pre-load the self-supervised CSV when called from Analyze mode
        preloaded_df = None
        if preloaded_csv:
            try:
                import pandas as _pd
                preloaded_df = _pd.read_csv(preloaded_csv)
                logger.info(f"Using self-supervised CSV for RF training: {preloaded_csv} ({len(preloaded_df)} rows)")
            except Exception as e:
                messagebox.showerror("Error", f"Could not load self-supervised CSV:\n{e}", parent=self.master)
                return
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

        self.start_button.configure(state="disabled")
        self.train_button.configure(state="disabled")
        from .header import TrainHeaderClassifierLogic
        training_gui = TrainClassifierGUI(self.master)
        self._active_training_gui = training_gui
        training_thread = threading.Thread(
            target=TrainHeaderClassifierLogic,
            args=(output_dir, training_gui, prompt_update_missing_fields),
            kwargs={"preloaded_df": preloaded_df},
            daemon=True
        )
        training_thread.start()
        self.master.after(100, self._poll_training_status, training_thread, training_gui)

    def train_classifier_cnn(self, preloaded_csv=None):
        if self.is_running: return
        output_dir = self.output_dir.get()
        try:
            import torch
        except Exception:
            logger.error("CNN training dependency not found (PyTorch)")
            messagebox.showerror("Error", "CNN training requires PyTorch (torch).", parent=self.master)
            return

        # Pre-load the self-supervised CSV when called from Analyze mode
        preloaded_df = None
        if preloaded_csv:
            try:
                import pandas as _pd
                preloaded_df = _pd.read_csv(preloaded_csv)
                logger.info(f"Using self-supervised CSV for CNN training: {preloaded_csv} ({len(preloaded_df)} rows)")
            except Exception as e:
                messagebox.showerror("Error", f"Could not load self-supervised CSV:\n{e}", parent=self.master)
                return

        logger.info("Starting Image-Based Model (CNN) training")
        self.start_button.configure(state="disabled")
        self.train_button.configure(state="disabled")
        training_gui = TrainClassifierGUI(self.master)
        self._active_training_gui = training_gui
        training_thread = threading.Thread(
            target=self._train_cnn_background,
            args=(output_dir, training_gui, preloaded_df),
            daemon=True
        )
        training_thread.start()
        self.master.after(100, self._poll_training_status, training_thread, training_gui)

    def _train_cnn_background(self, output_dir, training_gui, preloaded_df=None):
        # Delegate the full image-model training flow to the cnn module's TrainImageClassifierLogic
        try:
            from . import cnn as cnn_module
            cnn_module.TrainImageClassifierLogic(output_dir, training_gui, preloaded_df=preloaded_df)
        finally:
            # Ensure GUI is closed if the training function returns
            try:
                training_gui.close()
            except Exception:
                pass

    def _poll_training_status(self, thread, training_gui=None):
        if thread.is_alive():
            self.master.after(100, self._poll_training_status, thread, training_gui)
        else:
            # Training completed (or was cancelled)
            self.start_button.configure(state="normal")

            # Check if training was cancelled — skip auto-load and model check
            was_cancelled = training_gui is not None and training_gui.is_cancelled()
            if was_cancelled:
                self._check_model_status()
                return

            # Automatically load the freshly trained model(s)
            model_type = self.model_type.get()
            rf_needed = model_type in ["DICOM Header Model", "Header + Image Model"]
            cnn_needed = model_type in ["Image-Based Model", "Header + Image Model"]

            output_dir = self.output_dir.get()
            model_files_dir = os.path.join(output_dir, "model_files")
            rf_model_path = os.path.join(model_files_dir, "rf_classifier_model.joblib")
            cnn_model_path = os.path.join(model_files_dir, "cnn_classifier_model.pth")

            if rf_needed and os.path.exists(rf_model_path):
                self._load_classifier_and_encoder()

            if cnn_needed and os.path.exists(cnn_model_path):
                self._load_image_classifier()

            self._check_model_status()

    def start_classification(self):
        if self.is_running: return
        current_mode = self.classification_mode.get()

        if current_mode == "Analyze":
            self.start_analysis()
            return

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

        # Automatically load hierarchy.yaml and config.py from the output directory if present
        existing_hierarchy = None
        try:
            hierarchy_path = os.path.join(output_dir, "hierarchy.yaml")
            config_path = os.path.join(output_dir, "config.py")
            hierarchy_exists = os.path.exists(hierarchy_path)
            config_exists = os.path.exists(config_path)

            # Automatically load config if it exists; otherwise offer to copy the default
            if config_exists:
                try:
                    if self._load_config_from_file(config_path):
                        logger.info(f"Automatically loaded config from: {config_path}")
                    else:
                        logger.warning(f"Failed to load config from {config_path}, using current config")
                except Exception as e:
                    logger.warning(f"Failed to load config.py: {e} - using current config")
            else:
                # No config in output dir — a project config is required
                if not self._handle_missing_config(output_dir, config_path):
                    logger.info("Classification start aborted (no project config)")
                    return

            # Automatically load hierarchy if it exists
            if hierarchy_exists:
                try:
                    import yaml
                    with open(hierarchy_path, 'r') as hf:
                        existing_hierarchy = yaml.safe_load(hf)
                    logger.info("Automatically loaded existing hierarchy.yaml - skipping DICOM scan")
                except Exception as e:
                    logger.warning(f"Failed to load existing hierarchy.yaml: {e} - will rescan files")
                    existing_hierarchy = None
        except Exception:
            existing_hierarchy = None

        # Persist the current directories so they are the default on next launch
        _save_last_dirs(input_dir, output_dir)

        self.is_running = True
        self.stop_requested.clear()
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.train_button.configure(state="disabled")

        self.progress_frame = ctk.CTkFrame(self.master, fg_color="transparent")
        self.progress_frame.pack(padx=scale_value(20), pady=scale_value(10), fill='x')

        self.progress_label = tk.StringVar(value="Starting...")
        ctk.CTkLabel(self.progress_frame, textvariable=self.progress_label).pack(side="top")

        self.progress_bar = ThemedProgressBar(self.progress_frame, mode="determinate")
        self.progress_bar.pack(fill='x', pady=scale_value(5))

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
        self.stop_button.configure(text="Stopping...", state="disabled")

    def _poll_classification_status(self, thread):
        if thread.is_alive():
            self.master.after(100, self._poll_classification_status, thread)
        else:
            self.classification_finished(aborted=self.stop_requested.is_set())

    def classification_finished(self, aborted):
        # Idempotent: the worker thread calls this when it finishes, and the
        # status poller also calls it when it sees the thread end. Run the
        # teardown (and log) only once per workflow.
        if not self.is_running:
            return
        self.is_running = False
        self.stop_requested.clear()
        self.start_button.configure(state="normal")
        self.stop_button.configure(text="Stop", state="disabled")

        if aborted:
            logger.info("Classification workflow stopped by user")
        else:
            logger.info("Classification workflow completed successfully")

        if hasattr(self, 'progress_frame') and self.progress_frame.winfo_exists():
            self.progress_frame.destroy()
        self._check_model_status()  # handles train_button state including Analyze-mode rules

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
                        if not f.startswith(".") and os.path.splitext(f)[1] in ["", ".dcm", ".dicom", ".ima"]:
                            all_files.append(os.path.join(dcm_root, f))

                logger.info(f"Found {len(all_files)} files. Scanning for DICOM...")

                # 2. Setup progress window
                total_files = len(all_files)
                gui = ProgressGUI(total_files=total_files)

                # 3. Parallel processing (scan) — use batched workers to reduce per-file overhead
                pool = mp.Pool(get_num_parallel_workers())
                results = []

                try:
                    # Create chunks of files to process per worker call (batching reduces per-file overhead)
                    batch_size = 64
                    chunks = [all_files[i:i+batch_size] for i in range(0, len(all_files), batch_size)]

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

                # Hierarchy building and thumbnail generation report progress in the same ProgressGUI window
                organize_started = False
                thumbnail_started = False
                try:
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
                    accn = study_data.get("accn", "")
                    series_dict = study_data.get("series", {})
                    all_studies.append((mrn, study_uid, accn, series_dict))

            # Randomize study order for Manual/Semi-Auto modes if configured
            if mode in ["Manual", "Semi-Auto"] and getattr(config, 'RANDOMIZE_STUDY_ORDER', True):
                seed = getattr(config, 'STUDY_ORDER_SEED', 42)
                random.seed(seed)
                random.shuffle(all_studies)
                logger.info(f"Randomized study order using seed {seed}")

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
                classification_csv = os.path.join(
                    get_classified_outputs_dir(output_dir),
                    study_dir_name,
                    get_study_csv_name(study_dir_name)
                )
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
                        self.rf_field_types,
                        convert=self.convert_to_nifti.get(),
                        accn=accn
                    )
                else:
                    study_done_event = threading.Event()

                    # Run Semi-Auto model inference here on the background thread
                    # so it never blocks the GUI; Manual mode needs no inference.
                    optimized_result = None
                    if mode == "Semi-Auto":
                        optimized_result = processing.optimize_study_selections(
                            series_dict, model_type,
                            rf_model, rf_target_encoder, rf_feature_encoder,
                            cnn_model, cnn_target_encoder, self.rf_field_types
                        )

                    # Build the per-study GUI on the main thread (Tkinter is not
                    # thread-safe); this worker runs in a background thread.
                    def _show_classify(series_dict=series_dict, mrn=mrn,
                                       study_uid=study_uid, accn=accn,
                                       optimized_result=optimized_result):
                        classify_study_gui(
                            series_dict, mrn, study_uid, output_dir, self.stop_requested,
                            study_done_event, mode,
                            model_type, rf_model, rf_target_encoder, rf_feature_encoder,
                            cnn_model, cnn_target_encoder,
                            self.rf_field_types,
                            convert=self.convert_to_nifti.get(),
                            accn=accn,
                            optimized_result=optimized_result,
                        )
                    self.master.after(0, _show_classify)

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

        dcm2niix_path = shutil.which("dcm2niix")
        if not dcm2niix_path:
            logger.error("dcm2niix not found in PATH")
            messagebox.showerror(
                "Missing dcm2niix",
                "dcm2niix was not found in your PATH. Please install it to convert to NIfTI.",
                parent=self.master
            )
            return

        study_folders = []
        classified_dir = get_classified_outputs_dir(output_dir)
        if os.path.isdir(classified_dir):
            for folder in os.listdir(classified_dir):
                folder_path = os.path.join(classified_dir, folder)
                if os.path.isdir(folder_path):
                    study_id = extract_study_id_from_path(folder_path)
                    csv_path = os.path.join(folder_path, get_study_csv_name(study_id))
                    if os.path.exists(csv_path):
                        study_folders.append(folder_path)

        if not study_folders:
            messagebox.showinfo("No Studies", "No classified studies found in output directory.", parent=self.master)
            return

        progress_window = ctk.CTkToplevel(self.master)
        progress_window.title("Converting to NIfTI...")
        progress_window.geometry("500x150")
        cascade(progress_window, self.master)
        progress_window.protocol("WM_DELETE_WINDOW", lambda: None)  # Disable close

        status_label = ctk.CTkLabel(progress_window, text=f"Converting {len(study_folders)} studies...")
        status_label.pack(pady=scale_value(10))

        progress_bar = ThemedProgressBar(progress_window, length=400, mode="determinate", maximum=len(study_folders))
        progress_bar.pack(pady=scale_value(10))

        current_label = ctk.CTkLabel(progress_window, text="")
        current_label.pack(pady=scale_value(5))

        # Convert in background thread
        def convert_worker():
            converted_count = 0
            skipped_count = 0
            error_count = 0

            for idx, study_path in enumerate(study_folders):
                study_id = extract_study_id_from_path(study_path)

                progress_window.after(0, lambda i=idx, sid=study_id: current_label.configure(text=f"Processing: {sid}"))
                progress_window.after(0, lambda i=idx: progress_bar.config(value=i))

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

                        series_dir = os.path.dirname(path)
                        expected_nifti_name = get_nifti_name(study_id, classification)
                        expected_nifti_path = os.path.join(study_path, expected_nifti_name)

                        # Skip if the NIfTI already exists
                        if os.path.exists(expected_nifti_path):
                            skipped_count += 1
                            continue

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

            progress_window.after(0, lambda: progress_bar.config(value=len(study_folders)))
            progress_window.after(0, lambda: current_label.configure(text="Complete!"))

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

    def _export_dicoms_utility(self):
        """Utility to export all classified DICOMs to a user-specified location."""
        if self.is_running:
            messagebox.showwarning("Busy", "Cannot export DICOMs while classification is running.", parent=self.master)
            return

        output_dir = self.output_dir.get()
        if not os.path.isdir(output_dir):
            messagebox.showerror("Error", "Please specify a valid output directory.", parent=self.master)
            return

        export_dir = filedialog.askdirectory(
            title="Select Export Directory",
            parent=self.master
        )

        if not export_dir:
            logger.info("User cancelled DICOM export")
            return

        logger.info(f"Starting DICOM export to: {export_dir}")

        study_folders = []
        classified_dir = get_classified_outputs_dir(output_dir)
        if os.path.isdir(classified_dir):
            for folder in os.listdir(classified_dir):
                folder_path = os.path.join(classified_dir, folder)
                if os.path.isdir(folder_path):
                    study_id = extract_study_id_from_path(folder_path)
                    csv_path = os.path.join(folder_path, get_study_csv_name(study_id))
                    if os.path.exists(csv_path):
                        study_folders.append(folder_path)

        if not study_folders:
            messagebox.showinfo("No Studies", "No classified studies found in output directory.", parent=self.master)
            return

        progress_window = ctk.CTkToplevel(self.master)
        progress_window.title("Exporting DICOMs...")
        progress_window.geometry("500x150")
        cascade(progress_window, self.master)
        progress_window.protocol("WM_DELETE_WINDOW", lambda: None)  # Disable close

        status_label = ctk.CTkLabel(progress_window, text=f"Exporting DICOMs from {len(study_folders)} studies...")
        status_label.pack(pady=scale_value(10))

        progress_bar = ThemedProgressBar(progress_window, length=400, mode="determinate", maximum=len(study_folders))
        progress_bar.pack(pady=scale_value(10))

        current_label = ctk.CTkLabel(progress_window, text="")
        current_label.pack(pady=scale_value(5))

        # Export in background thread
        def export_worker():
            import csv
            total_files_copied = 0
            skipped_count = 0
            error_count = 0

            # Track processed series to avoid duplicates
            processed_series = set()

            for idx, study_path in enumerate(study_folders):
                study_id = extract_study_id_from_path(study_path)

                progress_window.after(0, lambda i=idx, sid=study_id: current_label.configure(text=f"Processing: {sid}"))
                progress_window.after(0, lambda i=idx: progress_bar.config(value=i))

                csv_path = os.path.join(study_path, get_study_csv_name(study_id))
                try:
                    with open(csv_path, 'r', newline='') as f:
                        reader = csv.DictReader(f)

                        for row in reader:
                            classification = str(row.get('Classification', '')).strip()
                            if classification.lower() in ['other', 'none', '']:
                                skipped_count += 1
                                continue

                            path = row.get('Path', '')
                            if not path or not os.path.exists(path):
                                logger.warning(f"DICOM path not found: {path}")
                                error_count += 1
                                continue

                            try:
                                ds = pydicom.dcmread(path, stop_before_pixels=True)
                                mrn = str(getattr(ds, 'PatientID', '')).strip()
                                study_uid = str(getattr(ds, 'StudyInstanceUID', '')).strip()
                                series_uid = str(getattr(ds, 'SeriesInstanceUID', '')).strip()

                                if not mrn or not study_uid or not series_uid:
                                    logger.warning(f"Missing metadata in DICOM file: {path}")
                                    error_count += 1
                                    continue

                                series_key = (mrn, study_uid, series_uid, classification)
                                if series_key in processed_series:
                                    continue
                                processed_series.add(series_key)

                                # Handle directory naming based on ALLOW_DUPLICATE_LABELS
                                # Spaces in the classification label are replaced with underscores
                                # so that the exported directory tree never contains spaces.
                                safe_classification = classification.replace(' ', '_')
                                if ALLOW_DUPLICATE_LABELS:
                                    # Use series_uid to make directory unique
                                    classified_series_dir = f"{safe_classification}-{series_uid}"
                                else:
                                    # Use just the label
                                    classified_series_dir = safe_classification

                                # Find all DICOMs in the same series directory
                                series_dir = os.path.dirname(path)

                                dicom_files = []
                                for file_name in os.listdir(series_dir):
                                    file_path = os.path.join(series_dir, file_name)
                                    if os.path.isfile(file_path):
                                        # Check if it's a DICOM file by trying to read it
                                        try:
                                            pydicom.dcmread(file_path, stop_before_pixels=True)
                                            dicom_files.append(file_path)
                                        except:
                                            # Not a DICOM file, skip
                                            pass

                                # Copy all DICOMs to export location
                                for dicom_file in dicom_files:
                                    ds_file = pydicom.dcmread(dicom_file, stop_before_pixels=True)
                                    instance_uid = str(getattr(ds_file, 'SOPInstanceUID', '')).strip()

                                    if not instance_uid:
                                        logger.warning(f"Missing SOPInstanceUID in {dicom_file}, using filename")
                                        instance_uid = os.path.splitext(os.path.basename(dicom_file))[0]

                                    # Build export path: output/{mrn}/{study_uid}/{classified_series_name}/{instance_uid}.dcm
                                    export_path = os.path.join(
                                        export_dir,
                                        mrn,
                                        study_uid,
                                        classified_series_dir,
                                        f"{instance_uid}.dcm"
                                    )

                                    os.makedirs(os.path.dirname(export_path), exist_ok=True)

                                    shutil.copy2(dicom_file, export_path)
                                    total_files_copied += 1

                            except Exception as e:
                                logger.error(f"Error exporting DICOMs from {path}: {e}")
                                error_count += 1

                except Exception as e:
                    logger.error(f"Error processing {csv_path}: {e}")
                    error_count += 1

            progress_window.after(0, lambda: progress_bar.config(value=len(study_folders)))
            progress_window.after(0, lambda: current_label.configure(text="Complete!"))

            summary = "DICOM Export Complete!\n\n"
            summary += f"Files copied: {total_files_copied}\n"
            summary += f"Series skipped (Other/None): {skipped_count}\n"
            if error_count > 0:
                summary += f"Errors: {error_count}"

            logger.info(f"DICOM export complete - Files copied: {total_files_copied}, Skipped: {skipped_count}, Errors: {error_count}")

            progress_window.after(0, lambda: messagebox.showinfo("Export Complete", summary, parent=self.master))
            progress_window.after(0, progress_window.destroy)

        thread = threading.Thread(target=export_worker, daemon=True)
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

        dialog = ctk.CTkToplevel(self.master)
        dialog.title("Erase Data")
        dialog.geometry("550x340")
        cascade(dialog, self.master)
        dialog.transient(self.master)
        dialog.grab_set()

        ctk.CTkLabel(dialog, text="Select data to erase:", font=ctk.CTkFont(size=13, weight="bold")).pack(pady=scale_value(10))

        delete_classifications = tk.BooleanVar(value=False)
        delete_models = tk.BooleanVar(value=False)
        delete_hierarchy = tk.BooleanVar(value=False)
        delete_config = tk.BooleanVar(value=False)

        ctk.CTkCheckBox(
            dialog,
            text="Delete Classifications (study folders, NIfTIs, CSVs)",
            variable=delete_classifications,
        ).pack(anchor="w", padx=scale_value(20), pady=scale_value(5))

        ctk.CTkCheckBox(
            dialog,
            text="Delete Models (trained models, encoders, results)",
            variable=delete_models,
        ).pack(anchor="w", padx=scale_value(20), pady=scale_value(5))

        ctk.CTkCheckBox(
            dialog,
            text="Delete Hierarchy (hierarchy.yaml, dicom_symlinks)",
            variable=delete_hierarchy,
        ).pack(anchor="w", padx=scale_value(20), pady=scale_value(5))

        ctk.CTkCheckBox(
            dialog,
            text="Delete Config (config.py saved in output directory)",
            variable=delete_config,
        ).pack(anchor="w", padx=scale_value(20), pady=scale_value(5))

        ctk.CTkLabel(
            dialog,
            text="Warning: This action cannot be undone!",
            text_color="#ff6b6b",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(pady=scale_value(10))

        button_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        button_frame.pack(pady=scale_value(10))

        def confirm_delete():
            if not any([delete_classifications.get(), delete_models.get(), delete_hierarchy.get(), delete_config.get()]):
                messagebox.showwarning("No Selection", "Please select at least one option to delete.", parent=dialog)
                return

            items = []
            if delete_classifications.get():
                items.append("- All classification study folders")
            if delete_models.get():
                items.append("- All trained models and training data")
            if delete_hierarchy.get():
                items.append("- Hierarchy and temporary files")
            if delete_config.get():
                items.append("- Saved configuration file (config.py)")

            confirm_msg = "Are you sure you want to DELETE:\n\n" + "\n".join(items) + "\n\nThis cannot be undone!"

            response = messagebox.askyesno("Confirm Deletion", confirm_msg, parent=dialog, icon='warning')
            if not response:
                logger.info("User cancelled data deletion")
                return

            logger.info(f"User confirmed data deletion - Classifications: {delete_classifications.get()}, Models: {delete_models.get()}, Hierarchy: {delete_hierarchy.get()}, Config: {delete_config.get()}")
            dialog.destroy()
            self._perform_data_deletion(delete_classifications.get(), delete_models.get(), delete_hierarchy.get(), delete_config.get())

        ctk.CTkButton(button_frame, text="Delete", command=confirm_delete, width=100,
                      fg_color=ui_theme.RED, hover_color=ui_theme.RED_HOVER,
                      font=ctk.CTkFont(weight="bold")).pack(side="left", padx=scale_value(5))
        ctk.CTkButton(button_frame, text="Cancel", command=dialog.destroy, width=100,
                      fg_color=ui_theme.NEUTRAL, hover_color=ui_theme.NEUTRAL_HOVER).pack(side="left", padx=scale_value(5))

    def _perform_data_deletion(self, delete_classifications, delete_models, delete_hierarchy, delete_config):
        """Perform the actual data deletion."""
        output_dir = self.output_dir.get()
        deleted_items = []
        errors = []

        try:
            # Delete classifications (study folders with CSVs)
            if delete_classifications:
                classified_dir = get_classified_outputs_dir(output_dir)
                if os.path.isdir(classified_dir):
                    for folder in os.listdir(classified_dir):
                        folder_path = os.path.join(classified_dir, folder)
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
                model_files_dir = os.path.join(output_dir, "model_files")

                if os.path.exists(model_files_dir):
                    try:
                        shutil.rmtree(model_files_dir)
                        deleted_items.append("Model files directory (all model files and data)")
                    except Exception as e:
                        errors.append(f"Failed to delete model_files directory: {e}")

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

            # Delete config
            if delete_config:
                config_path = os.path.join(output_dir, "config.py")

                if os.path.exists(config_path):
                    try:
                        os.remove(config_path)
                        deleted_items.append("config.py")
                        # Reset to repository config if we were using the deleted config
                        if self.active_config_path == config_path:
                            self.active_config_path = None
                            logger.info("Reverted to repository configuration after deleting custom config.py")
                    except Exception as e:
                        errors.append(f"Failed to delete config.py: {e}")

            if deleted_items or errors:
                # Create simplified messages based on what was deleted
                messages = []
                if delete_classifications and any('Study folder' in item for item in deleted_items):
                    messages.append("Successfully deleted classification data")
                if delete_models and any('Model files directory' in item for item in deleted_items):
                    messages.append("Successfully deleted model data")
                if delete_hierarchy and (any('hierarchy.yaml' in item for item in deleted_items) or
                                        any('dicom_symlinks' in item for item in deleted_items) or
                                        any('series_thumbnails' in item for item in deleted_items)):
                    messages.append("Successfully deleted hierarchy data")
                if delete_config and any('config.py' in item for item in deleted_items):
                    messages.append("Successfully deleted configuration file")

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

    def _edit_config_file(self):
        """Open the currently active config.py in the system's default text editor."""
        try:
            if self.active_config_path and os.path.exists(self.active_config_path):
                # Using a custom config that was loaded
                config_path = self.active_config_path
                config_source = "custom"
            else:
                # Using repository config
                config_path = os.path.join(os.path.dirname(__file__), 'config.py')
                config_source = "repository"

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

            if config_source == "custom":
                message = (
                    f"The currently active config file has been opened:\n{config_path}\n\n"
                    "After making changes, save the file and use 'Configuration > Reload Config File' to apply them.\n\n"
                    "Note: This is a custom config file. To edit the repository default, "
                    "reload the repository config first."
                )
            else:
                message = (
                    "The repository config file has been opened in your default text editor.\n\n"
                    "After making changes, save the file and use 'Configuration > Reload Config File' to apply them."
                )

            messagebox.showinfo(
                "Config File Opened",
                message,
                parent=self.master
            )

        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Failed to open config file:\n{e}",
                parent=self.master
            )

    def _reload_config(self, parent_dialog=None):
        """Reload the currently active config to apply any changes."""
        try:
            if self.active_config_path and os.path.exists(self.active_config_path):
                # Reload custom config file
                config_path = self.active_config_path
                if not self._load_config_from_file(config_path):
                    messagebox.showerror(
                        "Error",
                        f"Failed to reload config from:\n{config_path}",
                        parent=parent_dialog if parent_dialog else self.master
                    )
                    return
                logger.info(f"Reloaded custom config from {config_path}")
            else:
                # Reload repository config module
                importlib.reload(config)
                self.active_config_path = None
                logger.info("Reloaded repository config")

            # Re-import specific values that are used directly in this module
            # Note: We use config.VARIABLE_NAME elsewhere, so those will automatically update
            global TARGET_SERIES, NUM_COLUMNS, NUM_ROWS, ALLOW_DUPLICATE_LABELS, THUMBNAIL_SCALE_FACTOR
            TARGET_SERIES = sanitize_target_series(config.TARGET_SERIES)
            NUM_COLUMNS = config.NUM_COLUMNS
            NUM_ROWS = config.NUM_ROWS
            ALLOW_DUPLICATE_LABELS = config.ALLOW_DUPLICATE_LABELS
            THUMBNAIL_SCALE_FACTOR = config.THUMBNAIL_SCALE_FACTOR

            config_source = "custom" if self.active_config_path else "repository"
            messagebox.showinfo(
                "Config Reloaded",
                f"Configuration has been reloaded successfully from the {config_source} config file.\n\n"
                "Note: Some settings (loaded models, running processes, and certain UI elements) "
                "may require an application restart to fully take effect.",
                parent=parent_dialog if parent_dialog else self.master
            )

        except Exception as e:
            messagebox.showerror(
                "Reload Failed",
                f"Failed to reload configuration:\n\n{e}\n\n"
                "Please check the config file for syntax errors.",
                parent=parent_dialog if parent_dialog else self.master
            )

    def _reload_config_menu(self):
        """Wrapper for _reload_config when called from Configuration menu."""
        self._reload_config(parent_dialog=None)

    def _add_missing_params_menu(self):
        """Configuration-menu action: check the active config file for missing
        parameters and offer to add them with editable suggested defaults."""
        # Determine the active config file (custom/output config, else repo default)
        config_path = self.active_config_path
        if not config_path or not os.path.exists(config_path):
            output_config = os.path.join(self.output_dir.get(), "config.py") if self.output_dir.get() else None
            if output_config and os.path.exists(output_config):
                config_path = output_config
            else:
                config_path = config.__file__
                if config_path.endswith('.pyc'):
                    config_path = config_path[:-1]
        if not config_path or not os.path.exists(config_path):
            messagebox.showerror("Error", "No config file found to check.", parent=self.master)
            return

        success, missing_vars, _ = self._validate_config_file(config_path)
        if not missing_vars:
            messagebox.showinfo(
                "Config Up To Date",
                f"No missing parameters found in:\n{config_path}",
                parent=self.master
            )
            return
        self._prompt_add_missing_params(config_path, missing_vars)

    def _get_config_globals(self):
        """
        Extract all uppercase global variable names from the config module.
        Returns a list of variable names (excluding special variables like __name__).
        """
        globals_list = []
        for name in dir(config):
            # Include only uppercase variables that don't start with underscore
            if name.isupper() and not name.startswith('_'):
                globals_list.append(name)
        return globals_list

    def _validate_config_file(self, config_path):
        """
        Validate that a config file contains all required global variables.

        Args:
            config_path: Path to the config file to validate

        Returns:
            tuple: (success: bool, missing_vars: list, error_msg: str)
        """
        try:
            # Get expected globals from the repository config
            expected_globals = self._get_config_globals()

            import importlib.util
            spec = importlib.util.spec_from_file_location("temp_config", config_path)
            if spec is None or spec.loader is None:
                return False, [], "Could not load config file"

            temp_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(temp_module)

            missing_vars = []
            for var_name in expected_globals:
                if not hasattr(temp_module, var_name):
                    missing_vars.append(var_name)

            if missing_vars:
                return False, missing_vars, f"Missing required variables: {', '.join(missing_vars)}"

            return True, [], ""

        except SyntaxError as e:
            return False, [], f"Syntax error in config file: {e}"
        except Exception as e:
            return False, [], f"Error validating config file: {e}"

    def _load_config_from_file(self, config_path, prompt_missing=True):
        """
        Load config variables from a file and update the config module.

        Args:
            config_path: Path to the config file to load
            prompt_missing: If True and the file lacks newer parameters, offer to
                add them (with editable suggested defaults). Disabled on the
                internal reload after writing, to avoid recursion.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            success, missing_vars, error_msg = self._validate_config_file(config_path)
            if not success:
                if missing_vars:
                    # Config predates newer settings — load it and keep the
                    # current defaults for whatever it lacks.
                    logger.warning(
                        f"Config {config_path} is missing newer settings "
                        f"({', '.join(missing_vars)}); defaults will be used for those."
                    )
                else:
                    messagebox.showwarning(
                        "Invalid Config File",
                        f"The selected config file is invalid:\n\n{error_msg}\n\n"
                        "The original configuration will be used.",
                        parent=self.master
                    )
                    return False

            import importlib.util
            spec = importlib.util.spec_from_file_location("temp_config", config_path)
            temp_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(temp_module)

            config_globals = self._get_config_globals()

            for var_name in config_globals:
                if hasattr(temp_module, var_name):
                    setattr(config, var_name, getattr(temp_module, var_name))

            # Update local globals that are imported directly
            global TARGET_SERIES, NUM_COLUMNS, NUM_ROWS, ALLOW_DUPLICATE_LABELS, THUMBNAIL_SCALE_FACTOR
            TARGET_SERIES = sanitize_target_series(config.TARGET_SERIES)
            NUM_COLUMNS = config.NUM_COLUMNS
            NUM_ROWS = config.NUM_ROWS
            ALLOW_DUPLICATE_LABELS = config.ALLOW_DUPLICATE_LABELS
            THUMBNAIL_SCALE_FACTOR = config.THUMBNAIL_SCALE_FACTOR

            # Track that we're now using a custom config
            self.active_config_path = config_path

            logger.info(f"Successfully loaded config from: {config_path}")

            # Offer to add any parameters this (older) config file is missing.
            if prompt_missing and missing_vars:
                self._prompt_add_missing_params(config_path, missing_vars)
            return True

        except Exception as e:
            messagebox.showwarning(
                "Load Failed",
                f"Failed to load configuration:\n\n{e}\n\n"
                "The original configuration will be used.",
                parent=self.master
            )
            logger.error(f"Failed to load config from {config_path}: {e}")
            return False

    def _prompt_add_missing_params(self, config_path, missing_vars):
        """
        Prompt the user to add parameters that *config_path* is missing relative
        to the current MRID version, with editable suggested defaults. On accept,
        the chosen values are written into the config file and it is reloaded.
        """
        import ast

        # Suggested default for each missing param = the repository default
        # (config module still holds it, since the loaded file didn't set it).
        rows = []  # (name, default_repr)
        for name in missing_vars:
            try:
                rows.append((name, repr(getattr(config, name))))
            except Exception:
                rows.append((name, "None"))
        if not rows:
            return

        dlg = ctk.CTkToplevel(self.master)
        dlg.title("Add New Config Parameters")
        cascade(dlg, self.master)
        dlg.grab_set()

        ctk.CTkLabel(
            dlg,
            text=("This config file is missing parameters available in this "
                  "version of MRID.\nReview the suggested defaults below (edit "
                  "if desired) and click \"Add to Config\" to write them into:\n"
                  f"{config_path}"),
            justify="left", wraplength=scale_value(560)
        ).pack(padx=scale_value(16), pady=(scale_value(14), scale_value(8)), anchor="w")

        scroll = ctk.CTkScrollableFrame(dlg, width=scale_value(560), height=scale_value(260))
        scroll.pack(fill="both", expand=True, padx=scale_value(16))
        entry_vars = {}
        for r, (name, default_repr) in enumerate(rows):
            ctk.CTkLabel(scroll, text=name, anchor="w").grid(
                row=r, column=0, sticky="w", padx=scale_value(4), pady=scale_value(4))
            var = tk.StringVar(value=default_repr)
            ctk.CTkEntry(scroll, textvariable=var, width=scale_value(320)).grid(
                row=r, column=1, sticky="w", padx=scale_value(6), pady=scale_value(4))
            entry_vars[name] = var
        scroll.grid_columnconfigure(1, weight=1)

        btns = ctk.CTkFrame(dlg, fg_color="transparent")
        btns.pack(pady=scale_value(14))

        def on_add():
            # Validate every value is a safe Python literal before writing.
            replacements = {}
            for name, var in entry_vars.items():
                text = var.get().strip()
                try:
                    ast.literal_eval(text)
                except Exception as e:
                    messagebox.showwarning(
                        "Invalid Value",
                        f"The value for {name} is not a valid Python literal:\n\n{text}\n\n{e}",
                        parent=dlg
                    )
                    return
                replacements[name] = text
            try:
                with open(config_path, "r", encoding="utf-8") as fh:
                    source = fh.read()
                new_source = _rewrite_config_assignments(source, replacements)
                with open(config_path, "w", encoding="utf-8") as fh:
                    fh.write(new_source)
            except Exception as e:
                messagebox.showerror("Write Failed",
                                     f"Could not update the config file:\n{e}", parent=dlg)
                return
            dlg.grab_release()
            dlg.destroy()
            logger.info(f"Added missing parameters to {config_path}: {list(replacements)}")
            # Reload so the new values take effect (no re-prompt).
            self._load_config_from_file(config_path, prompt_missing=False)
            messagebox.showinfo(
                "Config Updated",
                f"Added {len(replacements)} parameter(s) to:\n{config_path}",
                parent=self.master
            )

        def on_skip():
            dlg.grab_release()
            dlg.destroy()

        ctk.CTkButton(btns, text="Add to Config", width=scale_value(140),
                      fg_color=ui_theme.GREEN, hover_color=ui_theme.GREEN_HOVER,
                      command=on_add).pack(side="left", padx=scale_value(6))
        ctk.CTkButton(btns, text="Skip", width=scale_value(100),
                      fg_color=ui_theme.NEUTRAL, hover_color=ui_theme.NEUTRAL_HOVER,
                      command=on_skip).pack(side="left", padx=scale_value(6))
        dlg.protocol("WM_DELETE_WINDOW", on_skip)
        dlg.wait_window()

    def _run_config_wizard(self):
        """Run the config wizard on the currently active config file."""
        if self.is_running:
            messagebox.showwarning("Busy", "Cannot run the config wizard while a workflow is running.",
                                   parent=self.master)
            return

        output_dir = self.output_dir.get()
        output_config = os.path.join(output_dir, "config.py") if output_dir else None

        # Choose the wizard's template (initial values) and save destination
        if self.active_config_path and os.path.exists(self.active_config_path):
            base_path = save_path = self.active_config_path
        elif output_config and os.path.exists(output_config):
            base_path = save_path = output_config
        elif output_dir and os.path.isdir(output_dir):
            base_path = None  # repository defaults
            save_path = output_config
        else:
            messagebox.showerror("Error", "Please specify a valid output directory first.",
                                 parent=self.master)
            return

        from .config_wizard import run_config_wizard
        saved = run_config_wizard(self.master, base_config_path=base_path, save_path=save_path,
                                  analyze_mode=(self.classification_mode.get() == "Analyze"))
        if saved and self._load_config_from_file(saved):
            self.active_config_path = saved
            self._check_model_status()
            messagebox.showinfo(
                "Config Saved",
                f"Configuration written and loaded:\n{saved}",
                parent=self.master
            )

    def _prompt_no_config(self):
        """
        Ask the user what to do when the output directory has no config.py.
        Returns 'wizard', 'copy', or 'cancel' (abort the workflow — a project
        config is required to proceed).
        """
        result = {"choice": "cancel"}

        dlg = ctk.CTkToplevel(self.master)
        dlg.title("No Config File Found")
        cascade(dlg, self.master)
        dlg.grab_set()
        dlg.resizable(False, False)

        ctk.CTkLabel(
            dlg,
            text=("No config.py was found in the output directory.\n\n"
                  "A project config file is required: run the Config Wizard to "
                  "set one up, or copy the default config file as-is.\n"
                  "Cancel to abort."),
            wraplength=scale_value(440), justify="left"
        ).pack(padx=scale_value(20), pady=scale_value(16))

        btns = ctk.CTkFrame(dlg, fg_color="transparent")
        btns.pack(pady=(0, scale_value(16)))

        def choose(choice):
            result["choice"] = choice
            dlg.grab_release()
            dlg.destroy()

        ctk.CTkButton(btns, text="Run Config Wizard", width=150,
                      fg_color=ui_theme.GREEN, hover_color=ui_theme.GREEN_HOVER,
                      command=lambda: choose("wizard")).pack(side="left", padx=scale_value(6))
        ctk.CTkButton(btns, text="Copy Default Config", width=150,
                      command=lambda: choose("copy")).pack(side="left", padx=scale_value(6))
        ctk.CTkButton(btns, text="Cancel", width=100,
                      fg_color=ui_theme.NEUTRAL, hover_color=ui_theme.NEUTRAL_HOVER,
                      command=lambda: choose("cancel")).pack(side="left", padx=scale_value(6))

        dlg.protocol("WM_DELETE_WINDOW", lambda: choose("cancel"))
        dlg.wait_window()
        return result["choice"]

    def _handle_missing_config(self, output_dir, config_path, analyze_mode=False):
        """
        Require a project config for an output directory that has none: offer
        the wizard or a copy of the default config. Returns True once a config
        exists and is loaded, False if the user cancelled (abort the workflow).
        """
        while True:
            choice = self._prompt_no_config()

            if choice == "wizard":
                from .config_wizard import run_config_wizard
                base = self.active_config_path if (self.active_config_path and
                                                   os.path.exists(self.active_config_path)) else None
                os.makedirs(output_dir, exist_ok=True)
                saved = run_config_wizard(self.master, base_config_path=base,
                                          save_path=config_path, analyze_mode=analyze_mode)
                if saved and self._load_config_from_file(saved):
                    self.active_config_path = saved
                    logger.info(f"Config wizard created and loaded: {saved}")
                    return True
                # Wizard cancelled — re-offer the choices
                continue
            elif choice == "copy":
                try:
                    if self.active_config_path and os.path.exists(self.active_config_path):
                        source_config_path = self.active_config_path
                    else:
                        source_config_path = config.__file__
                        if source_config_path.endswith('.pyc'):
                            source_config_path = source_config_path[:-1]
                    os.makedirs(output_dir, exist_ok=True)
                    shutil.copy(source_config_path, config_path)
                    logger.info(f"Copied default config to: {config_path}")
                    if self._load_config_from_file(config_path):
                        self.active_config_path = config_path
                    return True
                except Exception as e:
                    logger.warning(f"Failed to copy default config to output directory: {e}")
                    messagebox.showerror("Copy Failed",
                                         f"Could not copy the default config:\n{e}",
                                         parent=self.master)
                    continue
            else:
                logger.info("User cancelled config setup - aborting workflow start")
                return False

    def _save_config(self):
        """Save the current config.py file to the output directory and make it the active config."""
        output_dir = self.output_dir.get()

        if not os.path.isdir(output_dir):
            messagebox.showerror(
                "Error",
                "Please specify a valid output directory first.",
                parent=self.master
            )
            return

        try:
            if self.active_config_path and os.path.exists(self.active_config_path):
                # Save the currently active custom config
                source_config_path = self.active_config_path
            else:
                # Save the repository config
                source_config_path = config.__file__
                if source_config_path.endswith('.pyc'):
                    source_config_path = source_config_path[:-1]  # Remove 'c' to get .py

            dest_path = os.path.join(output_dir, "config.py")

            shutil.copy(source_config_path, dest_path)

            self.active_config_path = dest_path

            messagebox.showinfo(
                "Config Saved",
                f"Configuration saved successfully to:\n{dest_path}\n\n"
                f"This is now the active configuration for viewing and editing.",
                parent=self.master
            )
            logger.info(f"Config saved to: {dest_path} (now active)")

        except Exception as e:
            messagebox.showerror(
                "Save Failed",
                f"Failed to save configuration:\n\n{e}",
                parent=self.master
            )
            logger.error(f"Failed to save config: {e}")

    def _load_config(self):
        """Load a config.py file from a user-selected location."""
        config_path = filedialog.askopenfilename(
            title="Select Config File",
            initialdir=self.output_dir.get() if os.path.isdir(self.output_dir.get()) else None,
            filetypes=[("Python files", "*.py"), ("All files", "*.*")]
        )

        if not config_path:
            return  # User cancelled

        if self._load_config_from_file(config_path):
            messagebox.showinfo(
                "Config Loaded",
                f"Configuration loaded successfully from:\n{config_path}\n\n"
                "Note: Some settings (loaded models, running processes, and certain UI elements) "
                "may require an application restart to fully take effect.",
                parent=self.master
            )

    # =========================================================================
    # Analyze Mode
    # =========================================================================

    def start_analysis(self):
        """Entry point for Analyze mode — scan + thumbnail generation then cluster."""
        input_dir = self.input_dir.get()
        output_dir = self.output_dir.get()

        if not os.path.isdir(input_dir):
            messagebox.showerror("Error", "Please specify a valid input directory.", parent=self.master)
            return

        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            messagebox.showerror("Error", f"Cannot create output directory:\n{e}", parent=self.master)
            return

        # Config setup prompt (same logic as start_classification); the
        # target-series section is locked since Analyze mode determines it
        config_path = os.path.join(output_dir, "config.py")
        if not os.path.exists(config_path):
            if not self._handle_missing_config(output_dir, config_path, analyze_mode=True):
                logger.info("Analysis start aborted (no project config)")
                return

        # Check whether scanning has already been done
        hierarchy_path = os.path.join(output_dir, "hierarchy.yaml")
        thumbnail_dir = os.path.join(output_dir, "series_thumbnails")
        scan_needed = (
            not os.path.exists(hierarchy_path)
            or not os.path.isdir(thumbnail_dir)
            or len(os.listdir(thumbnail_dir)) == 0
        )

        if not scan_needed:
            logger.info("Existing hierarchy and thumbnails found — skipping scan.")

        # Persist the current directories so they are the default on next launch
        _save_last_dirs(input_dir, output_dir)

        # Lock UI
        self.is_running = True
        self.stop_requested.clear()
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.train_button.configure(state="disabled")

        self.progress_frame = ctk.CTkFrame(self.master, fg_color="transparent")
        self.progress_frame.pack(padx=scale_value(20), pady=scale_value(10), fill='x')
        self.progress_label = tk.StringVar(value="Starting analysis…")
        ctk.CTkLabel(self.progress_frame, textvariable=self.progress_label).pack(side="top")
        self.progress_bar = ThemedProgressBar(self.progress_frame, mode="indeterminate")
        self.progress_bar.pack(fill='x', pady=scale_value(5))
        self.progress_bar.start(10)

        analysis_thread = threading.Thread(
            target=self.run_analysis_main,
            args=(input_dir, output_dir, scan_needed),
            daemon=True
        )
        analysis_thread.start()
        self.master.after(100, self._poll_classification_status, analysis_thread)

    def run_analysis_main(self, input_dir, output_dir, scan_needed):
        """Background worker for Analyze mode."""
        try:
            import multiprocessing as mp
            import yaml as _yaml
            from .scanner import build_hierarchy
            from .utils import write_yaml_hierarchy
            from .analyze import run_analysis

            os.makedirs(output_dir, exist_ok=True)
            update_log_file_location(output_dir)

            hierarchy_path = os.path.join(output_dir, "hierarchy.yaml")
            thumbnail_dir = os.path.join(output_dir, "series_thumbnails")

            # ── Phase 1: Scan (if needed) ────────────────────────────────────
            if scan_needed:
                self.master.after(0, lambda: self.progress_label.set("Scanning DICOM files…"))

                all_files = []
                for dcm_root, _, files in os.walk(input_dir):
                    for f in files:
                        if not f.startswith(".") and os.path.splitext(f)[1] in ["", ".dcm", ".dicom", ".ima"]:
                            all_files.append(os.path.join(dcm_root, f))

                logger.info(f"Found {len(all_files)} candidate files.")
                total_files = len(all_files)
                gui = ProgressGUI(total_files=total_files, title="Analyze — Scanning…")

                pool = mp.Pool(get_num_parallel_workers())
                results = []
                try:
                    from .scanner import check_dicom_batch
                    batch_size = 64
                    chunks = [all_files[i:i + batch_size] for i in range(0, len(all_files), batch_size)]
                    processed = 0
                    for batch_res in pool.imap_unordered(check_dicom_batch, chunks):
                        if batch_res:
                            results.extend(batch_res)
                            processed += len(batch_res)
                        else:
                            processed += batch_size
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

                dicom_records = [r for r in results if r is not None]
                logger.info(f"Identified {len(dicom_records)} valid DICOM records.")

                groups = set((r['patient'], r['study_uid'], r['series_uid']) for r in dicom_records)
                total_groups = max(1, len(groups))

                organize_started = [False]
                thumbnail_started = [False]

                def _build_cb(n):
                    try:
                        if not organize_started[0]:
                            gui.show_organize_progress("Organizing series…", total_groups)
                            organize_started[0] = True
                        gui.update_organize_progress(n, total_groups)
                    except Exception:
                        pass

                def _thumb_cb(current, total):
                    try:
                        if not thumbnail_started[0]:
                            gui.show_thumbnail_progress("Generating thumbnails…", total)
                            thumbnail_started[0] = True
                        gui.update_thumbnail_progress(current, total)
                    except Exception:
                        pass

                hierarchy = build_hierarchy(
                    dicom_records, output_dir,
                    progress_callback=_build_cb,
                    thumbnail_progress_callback=_thumb_cb
                )
                try:
                    write_yaml_hierarchy(hierarchy, output_dir)
                except Exception as e:
                    logger.warning(f"Failed to write hierarchy.yaml: {e}")
                try:
                    gui.close()
                except Exception:
                    pass

            else:
                # Load existing hierarchy
                try:
                    with open(hierarchy_path, 'r') as hf:
                        hierarchy = _yaml.safe_load(hf)
                    logger.info("Loaded existing hierarchy.yaml")
                except Exception as e:
                    logger.warning(f"Could not load hierarchy.yaml: {e}; using empty hierarchy")
                    hierarchy = {}

            if self.stop_requested.is_set():
                self.classification_finished(aborted=True)
                return

            # ── Phase 2: Feature loading ──────────────────────────────────────
            from mrid_app import config as _cfg
            num_workers = get_num_parallel_workers()

            thumbnail_files = [
                f for f in os.listdir(thumbnail_dir)
                if os.path.splitext(f)[1].lower() in {'.jpg', '.jpeg', '.png'}
            ] if os.path.isdir(thumbnail_dir) else []
            total_thumbs = max(1, len(thumbnail_files))

            import time as _time

            # The first callback call initialises the bar (mode, maximum, value=0)
            # then immediately applies the real value.  This avoids any thread
            # synchronisation: the init and first update happen atomically in one
            # after() call, so the bar is never in the wrong mode when updated.
            _load_inited = [False]
            _load_highest = [0]
            _load_last_ui_time = [0.0]
            _LOAD_UI_INTERVAL = 0.05  # seconds — cap GUI update rate

            def _load_progress(current, total):
                try:
                    if current <= _load_highest[0]:
                        return
                    _load_highest[0] = current
                    now = _time.monotonic()
                    is_final = (current >= total)
                    if is_final or not _load_inited[0] or \
                            (now - _load_last_ui_time[0]) >= _LOAD_UI_INTERVAL:
                        _load_last_ui_time[0] = now
                        init = not _load_inited[0]
                        _load_inited[0] = True
                        v, t = current, total
                        def _update(_v=v, _t=t, _init=init):
                            if _init:
                                self.progress_bar.stop()
                                self.progress_bar.config(
                                    mode="determinate", maximum=_t, value=0)
                            self.progress_bar.config(value=_v)
                            self.progress_label.set(
                                f"Loading thumbnails… {_v}/{_t}")
                        self.master.after(0, _update)
                except Exception:
                    pass

            # ── Phase 3: Clustering ───────────────────────────────────────────
            coverage_threshold = getattr(_cfg, 'ANALYZE_COVERAGE_THRESHOLD', 0.5)

            # Stage labels: callback is called with current=0 (init) then
            # current=1/2/3 just BEFORE each stage starts, so the label is
            # always accurate while that stage is actually running.
            _stage_labels = {0: "Clustering — PCA…",
                             1: "Clustering — UMAP…",
                             2: "Clustering — HDBSCAN…"}

            def _cluster_progress(current, total):
                try:
                    label = _stage_labels.get(current, "Clustering…")
                    def _update(_l=label, _init=(current == 0)):
                        if _init:
                            self.progress_bar.stop()
                            self.progress_bar.config(mode="indeterminate", value=0)
                            self.progress_bar.start(10)
                        self.progress_label.set(_l)
                    self.master.after(0, _update)
                except Exception:
                    pass

            result = run_analysis(
                thumbnail_dir=thumbnail_dir,
                hierarchy=hierarchy,
                coverage_threshold=coverage_threshold,
                num_workers=num_workers,
                load_progress_cb=_load_progress,
                cluster_progress_cb=_cluster_progress,
            )

            if self.stop_requested.is_set():
                self.classification_finished(aborted=True)
                return

            # Stash for use by _write_selfsupervised_csv after the naming GUI
            self._last_analysis_result = result
            self._last_hierarchy = hierarchy

            # ── Phase 4: Show naming GUI on main thread ───────────────────────
            self.master.after(0, lambda: self.progress_label.set(
                f"Analysis complete — {result['k']} clusters found. Opening naming dialog…"))
            self.master.after(0, lambda: self._show_cluster_naming_gui(result, output_dir))

        except Exception as e:
            logger.exception(f"Error in analysis workflow: {e}")
            self.master.after(0, lambda: messagebox.showerror(
                "Analysis Error", f"An error occurred during analysis:\n\n{e}", parent=self.master))
            self.classification_finished(aborted=True)

    def _show_cluster_naming_gui(self, analysis_result, output_dir):
        """
        Modal window: show 4 best-example thumbnails per cluster, ask the user to
        name each cluster, tick 'Prefer 3D', and optionally exclude a cluster.
        On Apply write to config.  Called on the main thread.
        """
        k = analysis_result["k"]
        best_examples = analysis_result["best_examples"]
        representative_names = analysis_result.get("representative_names", [])

        # Guard against an unreasonable number of clusters: naming hundreds of
        # series types by hand is impractical, and rendering that many widgets
        # can exhaust the window system (X pixmap limits). This almost always
        # means ANALYZE_COVERAGE_THRESHOLD is set too low.
        _MAX_NAMING_CLUSTERS = 60
        if k > _MAX_NAMING_CLUSTERS:
            try:
                threshold = getattr(config, 'ANALYZE_COVERAGE_THRESHOLD', 0.1)
            except Exception:
                threshold = 0.1
            messagebox.showwarning(
                "Too Many Clusters",
                f"Analyze produced {k} clusters — too many to name individually.\n\n"
                f"This usually means ANALYZE_COVERAGE_THRESHOLD (currently "
                f"{threshold}) is too low. A cluster is kept only if it appears "
                f"in at least that fraction of exams, so raising the threshold "
                f"yields fewer, larger series-type clusters.\n\n"
                f"Increase ANALYZE_COVERAGE_THRESHOLD in the config and run "
                f"Analyze again.",
                parent=self.master
            )
            self.classification_finished(aborted=False)
            return

        win = ctk.CTkToplevel(self.master)
        win.title("Analyze — Name Your Series Types")
        cascade(win, self.master)
        win.grab_set()
        win.resizable(True, True)

        THUMB_PX = scale_value(192)  # display size per thumbnail
        PAD = scale_value(6)

        ctk.CTkLabel(
            win,
            text=(
                f"Found {k} series type cluster(s).  "
                "For each cluster, review the example thumbnails and enter a name.\n"
                "Tick 'Prefer 3D' if this series type should prefer 3D acquisitions. "
                "Uncheck 'Include' to exclude a cluster from TARGET_SERIES.\n"
                "Tip: give two or more clusters the exact same name to merge them "
                "into a single series type.\n"
                "Click Apply to save the names to TARGET_SERIES / PREFER_3D in config.py."
            ),
            wraplength=scale_value(1100),
            justify="left"
        ).grid(row=0, column=0, columnspan=2, padx=PAD * 2, pady=PAD * 2, sticky="w")

        # Scrollable canvas for clusters
        container = ctk.CTkFrame(win, fg_color="transparent")
        container.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=PAD, pady=PAD)
        win.grid_rowconfigure(1, weight=1)
        win.grid_columnconfigure(0, weight=1)

        canvas = tk.Canvas(container, highlightthickness=0, bg=ui_theme.PANEL_BG)
        scrollbar = ctk.CTkScrollbar(container, orientation="vertical", command=canvas.yview)
        scroll_frame = ctk.CTkFrame(canvas, fg_color="transparent")
        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Per-cluster widgets
        name_vars = []
        prefer3d_vars = []
        include_vars = []
        row_frames = []
        thumb_refs = []   # keep ImageTk references alive

        # Thumbnail filename -> DICOM path map so example thumbnails support
        # mouse-wheel series scrolling
        thumb_to_dicom = {}
        try:
            from .analyze import _build_thumbnail_to_dicom_map
            hierarchy = getattr(self, '_last_hierarchy', None)
            if hierarchy:
                thumb_to_dicom = _build_thumbnail_to_dicom_map(hierarchy)
        except Exception:
            pass

        # Member count per cluster (number of series assigned to it)
        member_counts = {}
        try:
            import numpy as _np
            _labels = _np.asarray(analysis_result.get("labels"))
            for c in range(k):
                member_counts[c] = int(_np.sum(_labels == c))
        except Exception:
            pass

        for c in range(k):
            count = member_counts.get(c)
            title = "Name your series"
            if count is not None:
                title += f"  ({count} member{'s' if count != 1 else ''})"
            row_frame = TitledFrame(scroll_frame, title=title)
            row_frame.pack(fill="x", padx=PAD, pady=PAD)
            row_frames.append(row_frame)

            # Thumbnail grid (up to 4) — each slot is a fixed-pixel Frame so
            # sizing is always in pixels whether or not an image is loaded.
            thumb_frame = ctk.CTkFrame(row_frame.content, fg_color="transparent")
            thumb_frame.pack(side="left", padx=PAD)
            example_paths = best_examples.get(c, [])
            cluster_refs = []
            for col_i, img_path in enumerate(example_paths[:4]):
                slot = tk.Frame(thumb_frame, width=THUMB_PX, height=THUMB_PX, bg=ui_theme.PANEL_BG)
                slot.grid(row=0, column=col_i, padx=2, pady=2)
                slot.grid_propagate(False)
                lbl = tk.Label(slot, bg=ui_theme.PANEL_BG)
                lbl.place(relwidth=1, relheight=1)
                try:
                    pil_img = Image.open(img_path).convert("RGB").resize(
                        (THUMB_PX, THUMB_PX), Image.LANCZOS
                    )
                    tk_img = ImageTk.PhotoImage(pil_img, master=lbl)
                    lbl.config(image=tk_img)
                    cluster_refs.append(tk_img)
                    # Mouse-wheel over the example scrolls through its series
                    dicom_path = thumb_to_dicom.get(os.path.basename(img_path))
                    if dicom_path and os.path.exists(dicom_path):
                        attach_series_scrolling(lbl, dicom_path, THUMB_PX)
                except Exception:
                    cluster_refs.append(None)
            # Pad remaining slots if fewer than 4
            for col_i in range(len(example_paths), 4):
                slot = tk.Frame(thumb_frame, width=THUMB_PX, height=THUMB_PX, bg=ui_theme.PANEL_BG)
                slot.grid(row=0, column=col_i, padx=2, pady=2)
                slot.grid_propagate(False)
                tk.Label(slot, bg=ui_theme.PANEL_BG, fg="#888888", text="N/A", anchor="center").place(relwidth=1, relheight=1)
            thumb_refs.append(cluster_refs)

            # Controls: name entry, prefer 3D, include toggle
            ctrl_frame = ctk.CTkFrame(row_frame.content, fg_color="transparent")
            ctrl_frame.pack(side="left", padx=PAD * 2, anchor="w")

            default_name = (representative_names[c]
                            if c < len(representative_names) and representative_names[c]
                            else f"Series{c + 1}")
            name_var = tk.StringVar(value=default_name)
            prefer3d_var = tk.BooleanVar(value=False)
            include_var = tk.BooleanVar(value=True)

            # Include checkbox – toggling dims/restores the row visually.
            # cf (this row's control frame) is passed in explicitly so the
            # closure operates on the correct row rather than capturing the
            # loop's final ctrl_frame.
            def _make_toggle(rf, cf, iv, ne):
                def _toggle(*_):
                    included = iv.get()
                    state = "normal" if included else "disabled"
                    rf.set_title_color("#e0e0e0" if included else "#777777")
                    for child in cf.winfo_children():
                        if getattr(child, '_is_include_cb', False):
                            continue  # keep the Include checkbox itself enabled
                        try:
                            child.configure(state=state)
                        except Exception:
                            pass
                    # shade the name entry red when excluded
                    try:
                        if included:
                            ne.configure(state="normal", border_color="#565b5e")
                        else:
                            ne.configure(state="disabled", border_color="#aa4444")
                    except Exception:
                        pass
                return _toggle

            include_cb = ctk.CTkCheckBox(
                ctrl_frame,
                text="Include",
                variable=include_var,
            )
            include_cb._is_include_cb = True
            include_cb.grid(row=0, column=0, columnspan=2, sticky="w", pady=2)

            ctk.CTkLabel(ctrl_frame, text="Series name:").grid(row=1, column=0, sticky="w", pady=2)
            name_entry = ctk.CTkEntry(ctrl_frame, textvariable=name_var, width=170)
            name_entry.grid(row=1, column=1, sticky="w", padx=PAD, pady=2)

            include_var.trace_add("write", _make_toggle(row_frame, ctrl_frame, include_var, name_entry))

            prefer3d_cb = ctk.CTkCheckBox(
                ctrl_frame,
                text="Prefer 3D acquisition",
                variable=prefer3d_var,
            )
            prefer3d_cb.grid(row=2, column=0, columnspan=2, sticky="w", pady=2)

            name_vars.append(name_var)
            prefer3d_vars.append(prefer3d_var)
            include_vars.append(include_var)

        btn_frame = ctk.CTkFrame(win, fg_color="transparent")
        btn_frame.grid(row=2, column=0, columnspan=2, pady=PAD * 2)

        def _on_apply():
            included = [i for i, v in enumerate(include_vars) if v.get()]
            if not included:
                messagebox.showwarning(
                    "No Clusters Included",
                    "At least one cluster must be included before applying.",
                    parent=win
                )
                return
            names = [name_vars[i].get().strip() for i in included]
            prefer3d = [prefer3d_vars[i].get() for i in included]
            # Validate: no empty names among included clusters
            if any(n == "" for n in names):
                messagebox.showwarning(
                    "Missing Name",
                    "Please provide a name for every included cluster before applying.",
                    parent=win
                )
                return
            # Clusters given the same name are combined into one series type.
            # Confirm so an accidental duplicate name doesn't silently merge them.
            from collections import OrderedDict
            groups = OrderedDict()
            for nm in names:
                groups[nm] = groups.get(nm, 0) + 1
            merged = {nm: count for nm, count in groups.items() if count > 1}
            if merged:
                summary = "\n".join(f"  • \"{nm}\": {count} clusters"
                                    for nm, count in merged.items())
                proceed = messagebox.askyesno(
                    "Combine Clusters",
                    "Clusters sharing the same name will be combined into a "
                    "single series type for training:\n\n" + summary +
                    "\n\nContinue?",
                    parent=win
                )
                if not proceed:
                    return
            win.destroy()
            self._write_analysis_to_config(names, prefer3d, output_dir)
            self._write_selfsupervised_csv(included, names, output_dir)
            self.classification_finished(aborted=False)

        def _on_cancel():
            win.destroy()
            self.classification_finished(aborted=False)

        ctk.CTkButton(btn_frame, text="Apply", width=110, command=_on_apply,
                      fg_color=ui_theme.GREEN, hover_color=ui_theme.GREEN_HOVER).pack(side="left", padx=PAD)
        ctk.CTkButton(btn_frame, text="Cancel", width=110, command=_on_cancel,
                      fg_color=ui_theme.NEUTRAL, hover_color=ui_theme.NEUTRAL_HOVER).pack(side="left", padx=PAD)

        # ── Size and position the window ──────────────────────────────────────
        def _fit_window():
            win.update_idletasks()
            screen_w = win.winfo_screenwidth()
            screen_h = win.winfo_screenheight()
            max_h = int(screen_h * 0.90)
            max_w = int(screen_w * 0.95)

            # Measure the true content height: the scrollable cluster rows plus
            # the header and button frames that live outside the canvas.
            # winfo_reqheight() on the scroll_frame gives the full unclipped
            # height of all cluster rows.
            content_h = (scroll_frame.winfo_reqheight()
                         + win.grid_slaves(row=0)[0].winfo_reqheight()  # header
                         + btn_frame.winfo_reqheight()                  # buttons
                         + PAD * 8)                                     # padding

            # Content width: scroll_frame natural width plus scrollbar
            content_w = (scroll_frame.winfo_reqwidth()
                         + scrollbar.winfo_reqwidth()
                         + PAD * 4)

            final_w = max(min(content_w, max_w), scale_value(900))
            final_h = min(content_h, max_h)

            # Place at a fixed offset from the main window's top-left corner
            OFFSET_X = scale_value(40)
            OFFSET_Y = scale_value(40)
            x = self.master.winfo_rootx() + OFFSET_X
            y = self.master.winfo_rooty() + OFFSET_Y

            # Clamp so the dialog doesn't spill off the right or bottom edge
            x = max(0, min(x, screen_w - final_w))
            y = max(0, min(y, screen_h - final_h))

            # wm_geometry: sizes are measured real pixels, bypass CTk scaling
            win.wm_geometry(f"{final_w}x{final_h}+{x}+{y}")
            win.wm_minsize(final_w, min(final_h, scale_value(400)))

        win.after(0, _fit_window)
        win.wait_window()

    def _write_analysis_to_config(self, names: list, prefer3d: list, output_dir: str):
        """
        Write the user-defined cluster names and 3D preferences into config.

        Updates config module in memory, then rewrites the config file using a
        robust AST-based approach: parse the file, replace/insert TARGET_SERIES
        and PREFER_3D assignments, and write back with original formatting preserved
        for all other content.
        """
        import ast
        import tokenize
        import io

        # ── 0. Merge clusters sharing a name, then append "Other" ─────────────
        # Clusters the user gave the same name are one series type: collapse to
        # unique names (preserving first-seen order) and OR their 3D preference.
        # Any cluster named "Other" is dropped here so we control the single
        # catch-all "Other" entry at the end.
        unique_names = []
        prefer3d_by_name = {}
        for name, p3d in zip(names, prefer3d):
            if name.strip().lower() == "other":
                continue
            if name not in prefer3d_by_name:
                unique_names.append(name)
                prefer3d_by_name[name] = bool(p3d)
            else:
                prefer3d_by_name[name] = prefer3d_by_name[name] or bool(p3d)

        names = unique_names + ["Other"]
        prefer3d_by_name["Other"] = False

        # ── 1. Update in-memory config ────────────────────────────────────────
        setattr(config, 'TARGET_SERIES', names)
        prefer3d_dict = {name: prefer3d_by_name[name] for name in names}
        setattr(config, 'PREFER_3D', prefer3d_dict)
        # Also update the module-level global used elsewhere in gui.py
        global TARGET_SERIES
        TARGET_SERIES = sanitize_target_series(names)
        logger.info(f"Updated TARGET_SERIES={names}, PREFER_3D={prefer3d_dict}")


        # ── 2. Determine which config file to update ──────────────────────────
        config_file_path = None
        output_config = os.path.join(output_dir, "config.py")
        if os.path.exists(output_config):
            config_file_path = output_config
        elif self.active_config_path and os.path.exists(self.active_config_path):
            config_file_path = self.active_config_path
        else:
            src = config.__file__
            if src.endswith('.pyc'):
                src = src[:-1]
            if os.path.exists(src):
                config_file_path = src

        if not config_file_path:
            logger.warning("Could not locate a writable config.py — changes saved in memory only.")
            return

        # ── 3. Robust file rewrite ────────────────────────────────────────────
        try:
            with open(config_file_path, 'r', encoding='utf-8') as fh:
                original_source = fh.read()

            new_source = _rewrite_config_assignments(
                original_source,
                {
                    'TARGET_SERIES': repr(names),
                    'PREFER_3D': _format_prefer3d_dict(prefer3d_dict),
                }
            )

            with open(config_file_path, 'w', encoding='utf-8') as fh:
                fh.write(new_source)

            logger.info(f"Config file updated: {config_file_path}")
        except Exception as e:
            logger.error(f"Failed to write config file: {e}")
            messagebox.showwarning(
                "Config Write Warning",
                f"Could not write config.py:\n{e}\n\n"
                "Changes are active in memory for this session only.",
                parent=self.master
            )

    def _write_selfsupervised_csv(self, included_cluster_indices: list, names: list, output_dir: str):
        """
        Write a self-supervised classification CSV to
        ``<output_dir>/model_files/selfsupervised_classification_data.csv``.

        Rows are the high-confidence (probability ≥ threshold) members of each
        *included* cluster.  The format mirrors classification_data.csv:
          Classification, Path, <DICOM header fields…>

        *included_cluster_indices*  — list of cluster int ids that the user kept
        *names*                     — parallel list of user-assigned names
        *output_dir*                — root output directory
        """
        try:
            import csv as _csv
            import pydicom as _pydicom
            import numpy as np

            from .analyze import _build_thumbnail_to_dicom_map, _HIGH_CONF_THRESHOLD
            from .config import DICOM_HEADER_FIELDS
            from .utils import safe_get_dicom_attribute

            result = getattr(self, '_last_analysis_result', None)
            hierarchy = getattr(self, '_last_hierarchy', None)

            if result is None:
                logger.warning("_write_selfsupervised_csv: no analysis result available, skipping.")
                return

            paths = result.get('paths', [])
            labels = result.get('labels', np.array([]))
            probabilities = result.get('probabilities', np.zeros(len(paths)))

            if len(paths) == 0:
                logger.warning("_write_selfsupervised_csv: paths list is empty, skipping.")
                return

            paths_arr = np.array(paths)
            labels_arr = np.array(labels)
            probs_arr = np.array(probabilities)

            # Build thumbnail → dicom_path map from the hierarchy
            thumb_to_dicom = _build_thumbnail_to_dicom_map(hierarchy) if hierarchy else {}

            # Build fieldnames to match classification_data.csv
            fieldnames = ["Classification", "Path"] + [
                display_name for display_name, _ in DICOM_HEADER_FIELDS
            ]

            # ── Phase 1: collect (cluster_name, dicom_path) pairs cheaply ────
            # This is fast (just dict lookups + os.path.exists), so no progress
            # bar needed here.
            candidates = []   # list of (cluster_name, dicom_path)
            for cluster_idx, cluster_name in zip(included_cluster_indices, names):
                if cluster_name.strip().lower() == "other":
                    continue

                cluster_mask = labels_arr == cluster_idx
                hc_mask = cluster_mask & (probs_arr >= _HIGH_CONF_THRESHOLD)
                candidate_indices = np.where(hc_mask)[0]
                if len(candidate_indices) == 0:
                    candidate_indices = np.where(cluster_mask)[0]

                for idx in candidate_indices:
                    thumb_basename = os.path.basename(paths_arr[idx])
                    dicom_path = thumb_to_dicom.get(thumb_basename)
                    if dicom_path and os.path.exists(dicom_path):
                        candidates.append((cluster_name, dicom_path))

            if not candidates:
                logger.warning("_write_selfsupervised_csv: no resolvable DICOM paths found.")
                messagebox.showwarning(
                    "Self-Supervised CSV",
                    "No DICOM paths could be resolved for the included clusters.\n"
                    "The self-supervised CSV was not written.\n\n"
                    "This can happen when the hierarchy was loaded from a previous run "
                    "and DICOM paths are no longer available.",
                    parent=self.master
                )
                return

            total_files = len(candidates)

            # ── Phase 2: read DICOM headers in a background thread with a ─────
            # modal progress window on the main thread.
            rows = []
            errors = [0]
            _done = threading.Event()

            # Progress window
            prog_win = ctk.CTkToplevel(self.master)
            prog_win.title("Retrieving DICOM data…")
            prog_win.resizable(False, False)
            prog_win.grab_set()
            prog_win.protocol("WM_DELETE_WINDOW", lambda: None)  # block close

            PAD = scale_value(8)
            prog_label_var = tk.StringVar(
                value=f"Retrieving data from DICOM files (0/{total_files})")
            ctk.CTkLabel(prog_win, textvariable=prog_label_var,
                         anchor="w").pack(padx=PAD * 2, pady=(PAD * 2, PAD), fill="x")
            prog_bar = ThemedProgressBar(prog_win, mode="determinate",
                                         maximum=total_files, value=0)
            prog_bar.pack(padx=PAD * 2, pady=(0, PAD * 2), fill="x")
            prog_win.update_idletasks()

            prog_win.geometry("420x90")
            cascade(prog_win, self.master, step=60)

            # Shared counter updated by background thread, read by poll callback
            _completed = [0]

            def _dicom_worker():
                for cluster_name, dicom_path in candidates:
                    try:
                        ds = _pydicom.dcmread(
                            dicom_path, stop_before_pixels=True, force=True)
                        row_dict = {"Classification": cluster_name,
                                    "Path": dicom_path}
                        for display_name, dicom_attr in DICOM_HEADER_FIELDS:
                            row_dict[display_name] = safe_get_dicom_attribute(
                                ds, dicom_attr)
                        rows.append([row_dict.get(f, "") for f in fieldnames])
                    except Exception as e:
                        logger.debug(f"Could not read DICOM {dicom_path}: {e}")
                        errors[0] += 1
                    finally:
                        _completed[0] += 1
                _done.set()

            worker = threading.Thread(target=_dicom_worker, daemon=True)
            worker.start()

            # Poll the progress on the main thread every 100 ms
            def _poll_progress():
                n = _completed[0]
                try:
                    prog_bar.config(value=n)
                    prog_label_var.set(
                        f"Retrieving data from DICOM files ({n}/{total_files})")
                except Exception:
                    pass
                if not _done.is_set():
                    prog_win.after(100, _poll_progress)
                else:
                    # Final update then finish
                    try:
                        prog_bar.config(value=total_files)
                        prog_label_var.set(
                            f"Retrieving data from DICOM files "
                            f"({total_files}/{total_files})")
                        prog_win.update_idletasks()
                    except Exception:
                        pass
                    prog_win.after(200, _finish)

            def _finish():
                try:
                    prog_win.grab_release()
                    prog_win.destroy()
                except Exception:
                    pass

                total_written = len(rows)

                if not rows:
                    logger.warning(
                        "_write_selfsupervised_csv: all DICOM reads failed "
                        f"({errors[0]} errors).")
                    messagebox.showwarning(
                        "Self-Supervised CSV",
                        "All DICOM header reads failed; the CSV was not written.\n"
                        f"({errors[0]} error(s) — check the log for details.)",
                        parent=self.master
                    )
                    return

                model_files_dir = os.path.join(output_dir, "model_files")
                os.makedirs(model_files_dir, exist_ok=True)
                csv_out = os.path.join(
                    model_files_dir, "selfsupervised_classification_data.csv")

                try:
                    with open(csv_out, "w", newline="", encoding="utf-8") as fh:
                        writer = _csv.writer(fh)
                        writer.writerow(fieldnames)
                        writer.writerows(rows)
                except Exception as e:
                    logger.error(f"Failed to write self-supervised CSV: {e}")
                    messagebox.showwarning(
                        "Self-Supervised CSV Warning",
                        f"Could not write selfsupervised_classification_data.csv:\n{e}",
                        parent=self.master
                    )
                    return

                logger.info(
                    f"Wrote self-supervised CSV: {csv_out} "
                    f"({total_written} rows, {errors[0]} skipped)"
                )
                # Unlock Train Model(s) button for Analyze mode
                self._analyze_csv_ready = True
                self._selfsupervised_csv_path = csv_out
                self._check_model_status()
                messagebox.showinfo(
                    "Analysis Complete",
                    f"Analysis complete — {total_written} labelled series saved.\n\n"
                    "The Train Model(s) button is now active. Select your desired model type "
                    "and click Train Model(s) to train using the self-supervised data.",
                    parent=self.master
                )

            prog_win.after(100, _poll_progress)

        except Exception as e:
            logger.error(f"_write_selfsupervised_csv failed: {e}", exc_info=True)
            messagebox.showwarning(
                "Self-Supervised CSV Warning",
                f"Could not write selfsupervised_classification_data.csv:\n{e}",
                parent=self.master
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

        # Persist the current directories so they are the default on next launch
        _save_last_dirs(self.input_dir.get(), output_dir)

        self.is_running = True
        self.stop_requested.clear()
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.train_button.configure(state="disabled")

        self.progress_frame = ctk.CTkFrame(self.master, fg_color="transparent")
        self.progress_frame.pack(padx=scale_value(20), pady=scale_value(10), fill='x')

        self.progress_label = tk.StringVar(value="Starting verification...")
        ctk.CTkLabel(self.progress_frame, textvariable=self.progress_label).pack(side="top")

        self.progress_bar = ThemedProgressBar(self.progress_frame, mode="determinate")
        self.progress_bar.pack(fill='x', pady=scale_value(5))

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

            classified_dir = get_classified_outputs_dir(output_dir)
            if os.path.isdir(classified_dir):
                for folder in os.listdir(classified_dir):
                    folder_path = os.path.join(classified_dir, folder)
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

                    studies_to_delete.update(incorrect_studies)

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



# Config rewrite helpers live in config_io; keep the old underscore names
# used throughout this module as aliases.
from .config_io import (format_prefer3d_dict as _format_prefer3d_dict,
                        rewrite_config_assignments as _rewrite_config_assignments,
                        replace_single_assignment as _replace_single_assignment)


# `processing` and `load_center_slice_fast` are provided as lazy bindings near
# the top of this module so importing it stays fast.


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


def _safe_grab(win):
    """Make *win* modal once it is actually viewable.

    Calling grab_set() before the window is mapped raises
    'grab failed: window not viewable' (notably with CustomTkinter's deferred
    deiconify on Linux). Wait for visibility first, and retry on the event loop
    if needed."""
    def _grab(attempt=0):
        try:
            if win.winfo_viewable():
                win.grab_set()
                return
        except Exception:
            return
        if attempt < 50:
            win.after(20, lambda: _grab(attempt + 1))
    try:
        win.update_idletasks()
    except Exception:
        pass
    _grab()


def _series_desc_font():
    """A scaled font for series-description labels (plain tk.Labels aren't
    scaled by CustomTkinter, so they look tiny on HiDPI without this)."""
    try:
        family = ctk.ThemeManager.theme["CTkFont"]["family"]
    except Exception:
        family = None
    return (family, max(9, scale_value(12)))


def _fit_window_to_screen(root, content_frame, button_frame, pad_x=0, pad_y=0):
    """
    Resize *root* so it is as large as its scrollable content requires, but
    never larger than the screen.  Also repositions the window so it stays
    fully on-screen.

    Because the content lives inside a Canvas (for scrolling), querying
    root.winfo_reqwidth/height returns only the canvas minimum, not the
    actual content size.  We therefore measure content_frame and button_frame
    directly and reconstruct the desired window size from those values.

    Parameters
    ----------
    root         : Toplevel window
    content_frame: the inner Frame whose required size drives the width/height
    button_frame : the static button row frame (always fully visible)
    pad_x, pad_y : padding applied around the canvas area (same values passed
                   to _make_scrollable_frame)
    """
    try:
        root.update_idletasks()

        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()

        # Content (thumbnail grid) required size
        content_w = content_frame.winfo_reqwidth()
        content_h = content_frame.winfo_reqheight()

        # Button row required size
        btn_h = button_frame.winfo_reqheight()

        # Scrollbar width — read from the actual scrollbar widget if possible,
        # otherwise fall back to a safe default
        scrollbar_w = scale_value(20)
        try:
            # The scrollbar is the second child of the outer frame (canvas sibling)
            outer = content_frame.master.master  # content_frame → canvas → outer
            for child in outer.winfo_children():
                if isinstance(child, ctk.CTkScrollbar):
                    child.update_idletasks()
                    scrollbar_w = child.winfo_reqwidth()
                    break
        except Exception:
            pass

        # Total desired window size:
        #   width  = content width + scrollbar + horizontal padding on both sides
        #   height = content height + button row + vertical padding on both sides
        desired_w = content_w + scrollbar_w + 2 * pad_x
        desired_h = content_h + btn_h + 2 * pad_y

        win_w = min(desired_w, screen_w)
        win_h = min(desired_h, screen_h)

        # Try to keep the current top-left position; clamp so the window fits
        try:
            cur_x = root.winfo_x()
            cur_y = root.winfo_y()
        except Exception:
            cur_x, cur_y = 0, 0

        x = max(0, min(cur_x, screen_w - win_w))
        y = max(0, min(cur_y, screen_h - win_h))

        # wm_geometry: sizes are measured real pixels, bypass CTk scaling
        root.wm_geometry(f"{win_w}x{win_h}+{x}+{y}")
        root.update_idletasks()
    except Exception:
        pass


def _make_scrollable_frame(root, pad_x=0, pad_y=0):
    """
    Replace the bare content frame with a Canvas + vertical Scrollbar.

    Returns (canvas, content_frame) where content_frame is the inner Frame
    that callers should populate with widgets.  The canvas occupies grid row 0
    of *root*; callers should place any static bottom widgets in row 1.

    Mouse-wheel scrolling is bound for Windows (<MouseWheel>), Linux
    (<Button-4> / <Button-5>), and macOS (<MouseWheel> with delta/120).
    """
    # Outer frame holds canvas + scrollbar side-by-side
    outer = ctk.CTkFrame(root, fg_color="transparent")
    outer.grid(row=0, column=0, padx=pad_x, pady=pad_y, sticky="nsew")
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)
    outer.grid_rowconfigure(0, weight=1)
    outer.grid_columnconfigure(0, weight=1)

    canvas = tk.Canvas(outer, highlightthickness=0, bg=ui_theme.PANEL_BG)
    canvas.grid(row=0, column=0, sticky="nsew")

    scrollbar = ctk.CTkScrollbar(outer, orientation="vertical", command=canvas.yview)
    scrollbar.grid(row=0, column=1, sticky="ns")
    canvas.configure(yscrollcommand=scrollbar.set)

    content_frame = tk.Frame(canvas, bg=ui_theme.PANEL_BG)
    canvas_window = canvas.create_window((0, 0), window=content_frame, anchor="nw")

    def _on_frame_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))

    def _on_canvas_configure(event):
        # Make the inner frame as wide as the canvas
        canvas.itemconfig(canvas_window, width=event.width)

    content_frame.bind("<Configure>", _on_frame_configure)
    canvas.bind("<Configure>", _on_canvas_configure)

    def _scroll(delta):
        canvas.yview_scroll(int(-1 * delta), "units")

    def _on_mousewheel(event):
        # Windows & macOS give event.delta in multiples of 120
        _scroll(event.delta / 120)

    def _on_button4(event):
        # Linux scroll up
        _scroll(1)

    def _on_button5(event):
        # Linux scroll down
        _scroll(-1)

    canvas.bind("<MouseWheel>", _on_mousewheel)
    canvas.bind("<Button-4>", _on_button4)
    canvas.bind("<Button-5>", _on_button5)
    content_frame.bind("<MouseWheel>", _on_mousewheel)
    content_frame.bind("<Button-4>", _on_button4)
    content_frame.bind("<Button-5>", _on_button5)

    return canvas, content_frame


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
    root = ctk.CTkToplevel()
    cascade(root, tk._default_root)

    batch_end = batch_start + len(batch_df)
    root.title(f"Verify {target_series} - Showing {batch_start+1}-{batch_end} of {total_series}")
    root.protocol("WM_DELETE_WINDOW", lambda: None)
    # grab_set is deferred to _safe_grab (after the window is sized and viewable)

    # Cap window height to screen height so the bottom buttons are always reachable
    screen_h = root.winfo_screenheight()
    root.maxsize(root.winfo_screenwidth(), screen_h)

    # Scrollable content area (canvas + scrollbar in row 0)
    _canvas, content_frame = _make_scrollable_frame(root, pad_x=scale_value(20), pad_y=scale_value(20))

    for i in range(NUM_COLUMNS):
        content_frame.grid_columnconfigure(i, weight=1)

    # Thumbnail pixel size (DPI-scaled + user scale factor)
    thumb_px = scale_thumbnail(256)

    loading_img = Image.new("RGB", (thumb_px, thumb_px), (45, 45, 45))
    tk_loading = ImageTk.PhotoImage(loading_img)

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

        series_frame = tk.Frame(content_frame, bg=ui_theme.PANEL_BG,
                                padx=scale_value(5), pady=scale_value(5))
        series_frame.grid(row=row_num, column=col_num, sticky="nsew")
        series_frame.grid_columnconfigure(0, weight=1)

        col_num += 1
        if col_num >= NUM_COLUMNS:
            col_num = 0
            row_num += 1

        # Image panel with loading indicator
        panel = tk.Label(series_frame, image=tk_loading, width=thumb_px, height=thumb_px,
                         bg=ui_theme.PANEL_BG)
        panel.image = tk_loading
        panel.grid(row=0, column=0, pady=(0, scale_value(5)))
        panel_widgets[series_path] = panel
        # Mouse-wheel over the thumbnail scrolls through the full series
        attach_series_scrolling(panel, series_path, thumb_px)

        tk.Label(series_frame, text=series_desc, wraplength=scale_thumbnail(250), justify="center",
                 bg=ui_theme.PANEL_BG, fg="#e0e0e0", font=_series_desc_font()).grid(
            row=1, column=0, pady=scale_value(5))

        # Toggle button for Correct/Incorrect
        toggle_var = tk.BooleanVar(value=False)  # False = Correct, True = Incorrect
        toggle_states[series_path] = (toggle_var, study_folder)

        def make_toggle_callback(var, btn, study_path, study_dir):
            def toggle():
                new_state = not var.get()
                var.set(new_state)
                if new_state:  # Incorrect
                    btn.configure(text="Incorrect", fg_color=ui_theme.RED, hover_color=ui_theme.RED_HOVER)
                else:  # Correct
                    btn.configure(text="Correct", fg_color=ui_theme.NEUTRAL, hover_color=ui_theme.NEUTRAL_HOVER)
            return toggle

        toggle_btn = ctk.CTkButton(
            series_frame,
            text="Correct",
            width=130,
            fg_color=ui_theme.NEUTRAL,
            hover_color=ui_theme.NEUTRAL_HOVER,
            command=None  # Will set below
        )
        toggle_btn.configure(command=make_toggle_callback(toggle_var, toggle_btn, series_path, study_folder))
        toggle_btn.grid(row=2, column=0, pady=(0, scale_value(5)))
        toggle_buttons[series_path] = toggle_btn

    # Load all images in parallel using ThreadPoolExecutor
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def load_single_thumbnail(series_path, panel, output_dir):
        """Load a single thumbnail, preferring a pre-generated thumbnail (JPEG/PNG)."""
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

            target_size = scale_thumbnail(256)
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

    def load_all_thumbnails():
        """Load all thumbnails in parallel."""
        with ThreadPoolExecutor(max_workers=get_num_parallel_workers()) as executor:
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

    threading.Thread(target=load_all_thumbnails, daemon=True).start()

    button_frame = ctk.CTkFrame(root, fg_color="transparent")
    button_frame.grid(row=1, column=0, sticky="ew", pady=scale_value(10))
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

    action_font = ctk.CTkFont(size=15, weight="bold")
    btn_stop = ctk.CTkButton(
        button_frame,
        fg_color=ui_theme.RED,
        hover_color=ui_theme.RED_HOVER,
        text="Stop All (a)",
        font=action_font,
        command=on_stop_all
    )
    btn_skip = ctk.CTkButton(
        button_frame,
        fg_color=ui_theme.YELLOW,
        hover_color=ui_theme.YELLOW_HOVER,
        text="Skip (s)",
        font=action_font,
        command=on_skip
    )
    btn_approve = ctk.CTkButton(
        button_frame,
        fg_color=ui_theme.GREEN,
        hover_color=ui_theme.GREEN_HOVER,
        text="Approve (d)",
        font=action_font,
        command=on_approve
    )

    btn_stop.grid(row=0, column=1, padx=scale_value(10))
    btn_skip.grid(row=0, column=2, padx=scale_value(10))
    btn_approve.grid(row=0, column=3, padx=scale_value(10))

    root.bind("<a>", lambda event: on_stop_all())
    root.bind("<s>", lambda event: on_skip())
    root.bind("<d>", lambda event: on_approve())

    _fit_window_to_screen(root, content_frame, button_frame,
                          pad_x=scale_value(20), pad_y=scale_value(20))
    _safe_grab(root)


def load_image_worker(dicom_paths, panel_label_ref, root):
    """Load a center slice (PIL Image) in a worker thread and schedule creation
    of the PhotoImage on the main thread. The worker only does PIL work; the
    PhotoImage is constructed on the Tk main thread to avoid invisible images.
    """
    target_size = scale_thumbnail(256)

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
                       cnn_target_encoder=None, rf_field_types=None, convert=True, accn="",
                       optimized_result=None):
    """
    Per-study classification GUI. Creates a grid of series thumbnails and dropdowns.
    Signals completion via study_done_event.

    This runs on the main (GUI) thread. For Semi-Auto, pass the precomputed
    *optimized_result* (from processing.optimize_study_selections) so the model
    inference happens on the background thread and never blocks the UI.
    """
    if stop_event.is_set():
        return

    root = ctk.CTkToplevel()
    root.title(f"Classify Study ({mode}): MRN {mrn} | Accn {accn}")

    # Classification window appears at a fixed cascade offset from the main window
    cascade(root, root.master)

    root.protocol("WM_DELETE_WINDOW", lambda: None)
    # grab_set is deferred to _safe_grab (after the window is sized and viewable)

    # Cap window height to screen height so the bottom buttons are always reachable
    screen_h = root.winfo_screenheight()
    root.maxsize(root.winfo_screenwidth(), screen_h)

    # Scrollable content area (canvas + scrollbar in row 0)
    _canvas, content_frame = _make_scrollable_frame(root, pad_x=scale_value(20), pad_y=scale_value(20))

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

    # Thumbnail pixel size (DPI-scaled + user scale factor)
    thumb_px = scale_thumbnail(256)

    placeholder_img = Image.new("RGB", (thumb_px, thumb_px), (45, 45, 45))
    tk_placeholder = ImageTk.PhotoImage(placeholder_img)

    optimized_assignments = {}
    predictions_by_path = {}
    if mode == "Semi-Auto":
        # Use predictions computed on the background thread when provided;
        # only fall back to inline inference (which blocks the UI) if not.
        if optimized_result is None:
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

    if mode == "Semi-Auto":
        logger.info(f"Semi-auto classifying exam {study_uid}...")

    for series_desc, dicom_paths in study_dict.items():
        series_frame = tk.Frame(content_frame, bg=ui_theme.PANEL_BG,
                                padx=scale_value(5), pady=scale_value(5))
        series_frame.grid(row=row_num, column=col_num, sticky="nsew")
        series_frame.grid_columnconfigure(0, weight=1)

        col_num += 1
        if col_num >= NUM_COLUMNS:
            col_num = 0
            row_num += 1

        panel = tk.Label(series_frame, image=tk_placeholder, width=thumb_px, height=thumb_px,
                         bg=ui_theme.PANEL_BG)
        panel.image = tk_placeholder
        panel.grid(row=0, column=0, pady=(0, scale_value(5)))
        tk.Label(series_frame, text=series_desc, wraplength=scale_thumbnail(250), justify="center",
                 bg=ui_theme.PANEL_BG, fg="#e0e0e0", font=_series_desc_font()).grid(
            row=1, column=0, pady=scale_value(5))

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
                # (use the raw initial_value key before sanitizing, as prob dicts
                # are keyed by whatever label the model encoder produced)
                if 'combined' in prob_info and initial_value in prob_info['combined']:
                    combined_p = prob_info['combined'][initial_value]
                if 'rf' in prob_info and initial_value in prob_info['rf']:
                    rf_p = prob_info['rf'][initial_value]
                if 'cnn' in prob_info and initial_value in prob_info['cnn']:
                    cnn_p = prob_info['cnn'][initial_value]
            except Exception:
                pass

            # Sanitize the label so it matches the dropdown options (which are
            # built from the sanitized TARGET_SERIES list).  Do this AFTER the
            # probability lookup above so the raw label is still available there.
            sanitized_candidates = sanitize_target_series([initial_value])
            sanitized_label = sanitized_candidates[0] if sanitized_candidates else "Other"
            # If the sanitized label is the auto-appended "Other" sentinel that
            # sanitize_target_series always adds, fall back to "Other" only if
            # the sanitized result genuinely maps to nothing useful.
            if sanitized_label in dropdown_values:
                initial_value = sanitized_label
            else:
                initial_value = "Other"

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

        # neutral colors for the un-assigned ("Other") state
        default_bg = ui_theme.NEUTRAL
        default_fg = '#e0e0e0'

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
                self.btn = ctk.CTkButton(parent, text=self.var.get(), width=170,
                                         fg_color=default_bg, hover_color=ui_theme.NEUTRAL_HOVER,
                                         text_color=default_fg)
                self.menu = tk.Menu(parent, tearoff=0)
                ui_theme.style_menu(self.menu)

                for opt in self.options:
                    def _mk(v=opt):
                        def _cmd():
                            try:
                                self.var.set(v)
                            except Exception:
                                pass
                            self._update_display(v)
                            # enforce uniqueness when duplicates are not allowed: clear this
                            # value from any other dropdowns that currently hold it
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
                # When ALLOW_DUPLICATE_LABELS is disabled, selecting a non-'Other'
                # label clears that label from all other dropdowns
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
                        self.btn.configure(fg_color=default_bg, text_color=default_fg,
                                           hover_color=ui_theme.NEUTRAL_HOVER, text=val)
                    except Exception:
                        try:
                            self.btn.configure(text=val)
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
                        self.btn.configure(fg_color=bg, text_color=fg, hover_color=bg, text=val)
                    else:
                        self.btn.configure(fg_color=default_bg, text_color=default_fg, text=val)
                    try:
                        self.btn.update_idletasks()
                    except Exception:
                        pass
                except Exception:
                    pass

        control_row = tk.Frame(series_frame, bg=ui_theme.PANEL_BG)
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
        # Mouse-wheel over the thumbnail scrolls through the full series
        attach_series_scrolling(panel, first_path, thumb_px)

        # If duplicate labels are not allowed and this dropdown's initial value
        # duplicates an existing assignment, reset it to 'Other'
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
        """Load a single thumbnail, preferring a pre-generated thumbnail (JPEG/PNG)."""
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
                    # Fall back to the slower full-series loader
                    pil_img = processing.load_center_slice(dicom_path_list)

            if pil_img is None:
                return None

            target_size = scale_thumbnail(256)
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

    def load_all_thumbnails():
        """Load all thumbnails in parallel."""
        with ThreadPoolExecutor(max_workers=get_num_parallel_workers()) as executor:
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

    threading.Thread(target=load_all_thumbnails, daemon=True).start()

    button_frame = ctk.CTkFrame(root, fg_color="transparent")
    button_frame.grid(row=1, column=0, sticky="ew", pady=scale_value(10))
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

    action_font = ctk.CTkFont(size=15, weight="bold")
    btn_stop = ctk.CTkButton(button_frame, fg_color=ui_theme.RED, hover_color=ui_theme.RED_HOVER,
                             text="Stop All (a)", font=action_font, command=on_stop)
    btn_skip = ctk.CTkButton(button_frame, fg_color=ui_theme.YELLOW, hover_color=ui_theme.YELLOW_HOVER,
                             text="Skip (s)", font=action_font, command=on_skip)
    btn_done = ctk.CTkButton(button_frame, fg_color=ui_theme.GREEN, hover_color=ui_theme.GREEN_HOVER,
                             text="Approve (d)", font=action_font, command=on_done)

    btn_stop.grid(row=0, column=1, padx=scale_value(10))
    btn_skip.grid(row=0, column=2, padx=scale_value(10))
    btn_done.grid(row=0, column=3, padx=scale_value(10))

    root.bind("<a>", lambda event: on_stop())
    root.bind("<s>", lambda event: on_skip())
    root.bind("<d>", lambda event: on_done())

    _fit_window_to_screen(root, content_frame, button_frame,
                          pad_x=scale_value(20), pad_y=scale_value(20))
    _safe_grab(root)
