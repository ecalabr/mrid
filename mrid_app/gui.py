import os
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from .config import DATA_DIR, OUT_DIR, TARGET_SERIES, NUM_COLUMNS
from .utils import write_yaml_hierarchy
from .header import TrainHeaderClassifierLogic
from .cnn import train_cnn_model_logic, prepare_cnn_input_data, SimpleCNN
from .prep import _preprocess_df, encode_text_features
import joblib
import pydicom
from PIL import Image, ImageTk
import shutil
import colorsys


# Small helper to enable Tk autoscaling
def enable_tkinter_autoscaling(root):
    try:
        dpi = root.winfo_fpixels('1i')
        scaling_factor = dpi / 72.0
        root.tk.call('tk', 'scaling', scaling_factor)
    except Exception:
        pass
    try:
        import tkinter.font as tkfont
        for name in ("TkDefaultFont", "TkTextFont", "TkFixedFont", "TkMenuFont",
                     "TkHeadingFont", "TkCaptionFont", "TkSmallCaptionFont",
                     "TkIconFont", "TkTooltipFont"):
            try:
                f = tkfont.nametofont(name)
                f.configure(size=int(f.cget("size") * 1.3))
            except Exception:
                continue
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
        self.root = tk.Toplevel(master)
        self.root.title("Model Training In Progress...")
        self.root.geometry("450x120")
        self.root.protocol("WM_DELETE_WINDOW", self.do_nothing)
        self.root.grab_set()
        self.label = tk.StringVar(value="Initializing training process...")
        tk.Label(self.root, textvariable=self.label).pack(pady=10)
        self.progress = ttk.Progressbar(self.root, orient="horizontal",
                                        length=400, mode="indeterminate")
        self.progress.pack(pady=5, padx=20)
        self.progress.start(10)
        self.root.update()

    def update_status(self, new_text):
        self.root.after(0, lambda: self.label.set(new_text))
        self.root.after(0, self.root.update_idletasks)

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
        try:
            self.root.title(title)
        except Exception:
            pass
        self.root.geometry("500x160")
        self.root.protocol("WM_DELETE_WINDOW", self.do_nothing)
        self.label = tk.Label(self.root, text=label_text)
        self.label.pack(pady=10)
        self.progress = ttk.Progressbar(self.root, orient="horizontal",
                                        length=400, mode="determinate",
                                        maximum=total_files)
        self.progress.pack(pady=10)
        self.root.update()

    def update(self, value):
        self.progress["value"] = value
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
        self.cnn_model = None
        self.cnn_target_encoder = None
        self.stop_requested = threading.Event()
        self.is_running = False
        # New flag for nifti conversion
        self.convert_to_nifti = tk.BooleanVar(value=False)

        # UI layout
        dir_frame = tk.LabelFrame(master, text="Directories", padx=10, pady=10)
        dir_frame.pack(padx=20, pady=10, fill="x")
        tk.Label(dir_frame, text="Input DIR:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        tk.Entry(dir_frame, textvariable=self.input_dir, width=50).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        tk.Button(dir_frame, text="Browse", command=lambda: self.browse_dir(self.input_dir, "Select INPUT directory")).grid(row=0, column=2, padx=5, pady=5)
        tk.Label(dir_frame, text="Output DIR:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        tk.Entry(dir_frame, textvariable=self.output_dir, width=50).grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        tk.Button(dir_frame, text="Browse", command=lambda: self.browse_dir(self.output_dir, "Select OUTPUT directory")).grid(row=1, column=2, padx=5, pady=5)
        dir_frame.columnconfigure(1, weight=1)

        control_frame = tk.Frame(master, padx=20, pady=10)
        control_frame.pack(fill="x")
        mode_frame = tk.Frame(control_frame)
        mode_frame.pack(side="right", padx=5)
        tk.Label(mode_frame, text="Mode:").pack(side="left", padx=5)
        mode_options = ["Manual", "Semi-Auto", "Full-Auto"]
        # Replaced ttk.Combobox with tk.OptionMenu. Keep using the same
        # StringVar (self.classification_mode) and call _check_model_status via
        # the OptionMenu 'command' so existing logic that relies on the
        # variable and event handling remains unchanged.
        mode_menu = tk.OptionMenu(mode_frame, self.classification_mode, *mode_options, command=self._check_model_status)
        try:
            mode_menu.config(width=12)
        except Exception:
            pass
        mode_menu.pack(side="left", padx=5)
        model_type_frame = tk.Frame(control_frame)
        model_type_frame.pack(side="right", padx=5)
        tk.Label(model_type_frame, text="Model Type:").pack(side="left", padx=5)
        model_options = ["DICOM Header Model", "Image-Based Model", "Header + Image Model"]
        # Replace model-type Combobox with OptionMenu; preserve self.model_type
        # StringVar and call _check_model_status on selection changes.
        model_menu = tk.OptionMenu(model_type_frame, self.model_type, *model_options, command=self._check_model_status)
        try:
            model_menu.config(width=20)
        except Exception:
            pass
        model_menu.pack(side="left", padx=5)

        # New convert toggle (Yes/No)
        convert_frame = tk.Frame(control_frame)
        convert_frame.pack(side="right", padx=10)
        tk.Label(convert_frame, text="Convert to NIfTI:").pack(side="left", padx=(0,5))

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

        self.start_button = tk.Button(control_frame, text="▶ Start Classification", font=("Arial", 12, "bold"), command=self.start_classification, state="normal")
        self.start_button.pack(side="left", padx=5)
        self.stop_button = tk.Button(control_frame, text="Stop", font=("Arial", 12), command=self.stop_classification, state="disabled")
        self.stop_button.pack(side="left", padx=5)
        self.train_button = tk.Button(control_frame, text="Train Model(s)", font=("Arial", 12), command=self._train_model)
        self.train_button.pack(side="right", padx=5)
        self.load_button = tk.Button(control_frame, text="Load Model(s)", font=("Arial", 12), command=self._load_model_handler)
        self.load_button.pack(side="right", padx=5)
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
                    print("dcm2niix was not found in PATH. Please install or add to PATH.")
                # revert toggle
                self.convert_to_nifti.set(False)
                try:
                    if hasattr(self, '_convert_button'):
                        self._convert_button.config(text="No")
                except Exception:
                    pass
                return

    def _load_model_handler(self):
        output_dir = self.output_dir.get()
        model_type = self.model_type.get()
        rf_needed = model_type in ["DICOM Header Model", "Header + Image Model"]
        cnn_needed = model_type in ["Image-Based Model", "Header + Image Model"]
        rf_model_path = os.path.join(output_dir, "rf_classifier_model.joblib")
        if rf_needed:
            if os.path.exists(rf_model_path):
                self._load_classifier_and_encoder()
            else:
                self.train_classifier_rf()
        cnn_model_path = os.path.join(output_dir, "cnn_classifier_model.pth")
        if cnn_needed:
            if os.path.exists(cnn_model_path):
                self._load_image_classifier()
            else:
                self.train_classifier_cnn()
        self._check_model_status()

    def _load_classifier_and_encoder(self):
        output_dir = self.output_dir.get()
        model_path = os.path.join(output_dir, "rf_classifier_model.joblib")
        target_encoder_path = os.path.join(output_dir, "label_encoder.joblib")
        feature_encoder_path = os.path.join(output_dir, "feature_encoder.joblib")
        if not os.path.exists(model_path) or not os.path.exists(target_encoder_path) or not os.path.exists(feature_encoder_path):
            messagebox.showwarning("Load Warning", "Header Model files not found. Skipping RF load.")
            return
        try:
            self.rf_model = joblib.load(model_path)
            self.rf_target_encoder = joblib.load(target_encoder_path)
            self.rf_feature_encoder = joblib.load(feature_encoder_path)
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load RF model/encoders: {e}")
            self.rf_model = None
            self.rf_target_encoder = None
            self.rf_feature_encoder = None

    def _load_image_classifier(self):
        try:
            import torch
            if torch is None:
                messagebox.showwarning("Load Warning", "PyTorch is not installed. Skipping CNN load.")
                return
        except Exception:
            messagebox.showwarning("Load Warning", "PyTorch is not installed. Skipping CNN load.")
            return
        output_dir = self.output_dir.get()
        model_path = os.path.join(output_dir, "cnn_classifier_model.pth")
        target_encoder_path = os.path.join(output_dir, "label_encoder.joblib")
        if not os.path.exists(model_path) or not os.path.exists(target_encoder_path):
            messagebox.showwarning("Load Warning", "Image Model files not found. Skipping CNN load.")
            return
        try:
            from .utils import warn_if_no_gpu_for_cnn
            # Show a warning/confirmation dialog if no GPU is available. If the user chooses not
            # to proceed (or PyTorch missing), skip loading.
            if not warn_if_no_gpu_for_cnn(gui_parent=self.master):
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
            state_dict = torch.load(model_path, map_location=map_loc)
            self.cnn_model.load_state_dict(state_dict)
            # Move model to the appropriate device
            try:
                self.cnn_model.to(map_loc)
            except Exception:
                pass
            self.cnn_model.eval()
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load CNN model: {e}")
            self.cnn_model = None
            self.cnn_target_encoder = None

    def _check_model_status(self, event=None):
        model_type = self.model_type.get()
        rf_loaded = self.rf_model is not None and self.rf_target_encoder is not None
        cnn_loaded = self.cnn_model is not None and self.cnn_target_encoder is not None
        can_start = False
        models_loaded = False
        mode_is_manual = False
        if self.classification_mode.get() == "Manual":
            can_start = True
            mode_is_manual = True
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
        if can_start:
            self.start_button.config(state="normal", bg="#4fff5b", activebackground="#4fff5b")
        else:
            self.start_button.config(state="disabled", bg="#f0f0f0", activebackground="#f0f0f0")
        if models_loaded:
            self.load_button.config(text="Model(s) Loaded", state="disabled", bg="#4fff5b", activebackground="#4fff5b")
        elif mode_is_manual:
            self.load_button.config(text="Load Model(s)", state="disabled", bg="#f0f0f0", activebackground="#f0f0f0")
        else:
            self.load_button.config(text="Load Model(s)", state="normal", bg="#ff4f4f", activebackground="#ff4f4f")
        train_available = True
        self.train_button.config(state="normal" if train_available else "disabled")

    def _train_model(self):
        if self.is_running:
            messagebox.showwarning("Busy", "Cannot start training while classification is running.")
            return
        model_type = self.model_type.get()
        rf_needed = model_type in ["DICOM Header Model", "Header + Image Model"]
        cnn_needed = model_type in ["Image-Based Model", "Header + Image Model"]
        if not rf_needed and not cnn_needed:
            messagebox.showinfo("Info", "No model type selected for training. Please choose a Model Type.")
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
            messagebox.showerror("Error", "Missing Dependencies", "RF training requires pandas, scikit-learn, and joblib.")
            return
        if not os.path.isdir(output_dir):
            messagebox.showerror("Error", "Invalid Output Directory", "Please specify a valid output directory containing classified data.")
            return
        self.start_button.config(state="disabled")
        self.train_button.config(state="disabled")
        training_gui = TrainClassifierGUI(self.master)
        training_thread = threading.Thread(
            target=TrainHeaderClassifierLogic,
            args=(output_dir, training_gui),
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
            messagebox.showerror("Error", "Missing PyTorch", "CNN training requires PyTorch (torch).")
            return
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
            self.start_button.config(state="normal")
            self._check_model_status()

    def start_classification(self):
        if self.is_running: return
        current_mode = self.classification_mode.get()
        model_type = self.model_type.get()
        if current_mode in ["Semi-Auto", "Full-Auto"]:
            rf_needed = model_type in ["DICOM Header Model", "Header + Image Model"]
            cnn_needed = model_type in ["Image-Based Model", "Header + Image Model"]
            rf_loaded = self.rf_model is not None and self.rf_target_encoder is not None
            cnn_loaded = self.cnn_model is not None and self.cnn_target_encoder is not None
            if (rf_needed and not rf_loaded) or (cnn_needed and not cnn_loaded):
                messagebox.showerror("Model Error", "Cannot start classification. Required model(s) are not loaded.")
                return
        input_dir = self.input_dir.get()
        output_dir = self.output_dir.get()
        if not os.path.isdir(input_dir):
            messagebox.showerror("Error", "Invalid Input Directory", "Please specify a valid input directory.")
            return

        # New: if hierarchy.yaml exists in output_dir, ask user whether to load it or rescan
        existing_hierarchy = None
        try:
            hierarchy_path = os.path.join(output_dir, "hierarchy.yaml")
            if os.path.exists(hierarchy_path):
                load_choice = messagebox.askyesno(
                    "Existing hierarchy found",
                    "A saved 'hierarchy.yaml' was found in the output directory.\n\n"
                    "Choose 'Yes' to load the existing hierarchy and skip scanning files,\n"
                    "or 'No' to rescan all files.")
                if load_choice:
                    try:
                        import yaml
                        with open(hierarchy_path, 'r') as hf:
                            existing_hierarchy = yaml.safe_load(hf)
                    except Exception as e:
                        messagebox.showwarning("Load Warning", f"Failed to load existing hierarchy.yaml: {e}\nWill rescan files instead.")
                        existing_hierarchy = None
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

                print(f"Found {len(all_files)} files. Scanning for DICOM...")

                # 2. Setup progress window
                gui = ProgressGUI(total_files=len(all_files))

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
                        if processed > len(all_files):
                            processed = len(all_files)
                        gui.update(processed)

                        if self.stop_requested.is_set():
                            gui.close()
                            pool.terminate()
                            pool.join()
                            self.classification_finished(aborted=True)
                            return
                finally:
                    gui.close()
                    pool.close()
                    pool.join()

                # 4. Process results
                dicom_records = [r for r in results if r is not None]
                print(f"Identified {len(dicom_records)} valid DICOM files.")

                # Show a progress window while building the hierarchy: count unique groups
                try:
                    groups = set((r['patient'], r['study_uid'], r['series_uid']) for r in dicom_records)
                    total_groups = max(1, len(groups))
                except Exception:
                    total_groups = max(1, len(dicom_records))

                build_gui = None
                try:
                    build_gui = ProgressGUI(total_files=total_groups, title="Building hierarchy...", label_text="Organizing series/groups...")
                    # pass a callback that updates the progress GUI
                    def _build_progress_cb(n):
                        try:
                            if build_gui:
                                build_gui.update(n)
                        except Exception:
                            pass

                    hierarchy = build_hierarchy(dicom_records, output_dir, progress_callback=_build_progress_cb)
                    # save hierarchy to YAML
                    try:
                        write_yaml_hierarchy(hierarchy, output_dir)
                    except Exception as e:
                        print(f"Failed to write hierarchy.yaml: {e}")
                finally:
                    try:
                        if build_gui:
                            build_gui.close()
                    except Exception:
                        pass

            # At this point `hierarchy` is available (either loaded or newly constructed)
            # 5. Loop through each study and classify
            all_studies = []
            for mrn, studies in hierarchy.items():
                for study_uid, series_dict in studies.items():
                    all_studies.append((mrn, study_uid, series_dict))

            total_studies = len(all_studies)
            try:
                self.progress_bar.config(maximum=total_studies)
            except Exception:
                pass

            model_type = self.model_type.get()
            rf_model, rf_target_encoder, rf_feature_encoder = self.rf_model, self.rf_target_encoder, self.rf_feature_encoder
            cnn_model, cnn_target_encoder = self.cnn_model, self.cnn_target_encoder

            for i, (mrn, study_uid, series_dict) in enumerate(all_studies):
                if self.stop_requested.is_set():
                    print("Classification loop aborted by user.")
                    self.classification_finished(aborted=True)
                    return

                self.progress_label.set(
                    f"Classifying Study {i + 1} of {total_studies} ({mode}/{model_type}): {mrn} / {study_uid}")

                classification_csv = os.path.join(output_dir, f"{mrn}-{study_uid}", f"{mrn}-{study_uid}_classification-results.csv")
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
                        convert=self.convert_to_nifti.get()
                    )
                else:
                    study_done_event = threading.Event()

                    # Call the per-study GUI (defined below in this module)
                    classify_study_gui(
                        series_dict, mrn, study_uid, output_dir, self.stop_requested,
                        study_done_event, mode,
                        model_type, rf_model, rf_target_encoder, rf_feature_encoder, cnn_model, cnn_target_encoder,
                        convert=self.convert_to_nifti.get()
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
            print(f"An error occurred in the classification process: {e}")
            self.classification_finished(aborted=True)


# At end of file, add asynchronous image loader and per-study classification GUI
from . import processing


def load_image_worker(dicom_paths, panel_label_ref, root):
    """Load a center slice (PIL Image) in a worker thread and schedule creation
    of the PhotoImage on the main thread. The worker only does PIL work; the
    PhotoImage is constructed on the Tk main thread to avoid invisible images.
    """
    try:
        pil_img = processing.load_center_slice(dicom_paths)
    except Exception:
        pil_img = None

    if pil_img is None:
        pil_img = Image.new('L', (256, 256), 128)

    try:
        pil_img = pil_img.resize((256, 256))
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
                       cnn_target_encoder=None, convert=True):
    """
    Per-study classification GUI. Creates a grid of series thumbnails and dropdowns.
    Signals completion via study_done_event.
    """
    if stop_event.is_set():
        return

    root = tk.Toplevel()
    root.title(f"Classify Study ({mode}): MRN {mrn} | Study {study_uid}")
    root.protocol("WM_DELETE_WINDOW", lambda: None)
    root.grab_set()

    content_frame = tk.Frame(root)
    content_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)

    for i in range(NUM_COLUMNS):
        content_frame.grid_columnconfigure(i, weight=1)

    dropdown_values = TARGET_SERIES
    dropdown_vars = {}

    # Precompute color map for the dropdown labels
    try:
        color_map = _generate_color_map(dropdown_values)
    except Exception:
        color_map = {k: (None, None) for k in dropdown_values}

    placeholder_img = Image.new("RGB", (256, 256), (200, 200, 200))
    tk_placeholder = ImageTk.PhotoImage(placeholder_img)

    optimized_assignments = {}
    predictions_by_path = {}
    if mode == "Semi-Auto":
        optimized_result = processing.optimize_study_selections(
            study_dict,
            model_type,
            rf_model, rf_target_encoder, rf_feature_encoder,
            cnn_model, cnn_target_encoder
        )
        # optimize_study_selections now returns (final_assignments, predictions_by_path)
        try:
            optimized_assignments, predictions_by_path = optimized_result
        except Exception:
            # backward compatibility: if an older version returned just dict
            optimized_assignments = optimized_result if optimized_result else {}
            predictions_by_path = {}
        if not optimized_assignments:
            print("Warning: Optimization failed, falling back to 'Other'.")

    row_num = 0
    col_num = 0

    for series_desc, dicom_paths in study_dict.items():
        series_frame = tk.Frame(content_frame, padx=5, pady=5)
        series_frame.grid(row=row_num, column=col_num, sticky="nsew")
        series_frame.grid_columnconfigure(0, weight=1)

        col_num += 1
        if col_num >= NUM_COLUMNS:
            col_num = 0
            row_num += 1

        panel = tk.Label(series_frame, image=tk_placeholder, width=256, height=256)
        panel.image = tk_placeholder
        panel.grid(row=0, column=0, pady=(0, 5))
        tk.Label(series_frame, text=series_desc, wraplength=250, justify="center").grid(row=1, column=0, pady=5)

        first_path = dicom_paths[0]
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
            print(f"  -> Series '{series_desc}' semi-auto pre-selected as: {initial_value}{prob_str}")
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
                self.btn = tk.Button(parent, text=self.var.get(), width=20, relief='raised')
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
                # Only enforce uniqueness when in Semi-Auto mode and a non-'Other' value was selected
                if not self.all_vars or self.mode != 'Semi-Auto' or val == 'Other':
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
            control_row.grid(row=2, column=0, pady=(0, 5))
        except Exception:
            try:
                control_row.pack(pady=(0,5))
            except Exception:
                pass

        try:
            cd._update_display()
        except Exception:
            pass

        dropdown_vars[first_path] = variable

        # Enforce uniqueness for initial semi-auto assignments: if this dropdown's
        # initial value duplicates an existing assignment (earlier dropdowns), reset it to 'Other'.
        try:
            if mode == 'Semi-Auto' and variable.get() != 'Other':
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

        # Start background image loader to replace placeholder with center slice
        try:
            threading.Thread(target=load_image_worker, args=(dicom_paths, panel, root), daemon=True).start()
        except Exception:
            pass

    button_frame = tk.Frame(root, pady=10)
    button_frame.grid(row=1, column=0, sticky="ew")
    button_frame.grid_columnconfigure(0, weight=1)
    button_frame.grid_columnconfigure(1, weight=0)
    button_frame.grid_columnconfigure(2, weight=0)
    button_frame.grid_columnconfigure(3, weight=0)
    button_frame.grid_columnconfigure(4, weight=1)

    def on_done():
        worker_args = (mrn, study_uid, output_root, dropdown_vars, study_dict)
        seen = set()
        duplicates = set()
        for item in dropdown_vars.values():
            if item.get() in seen and item.get() != "Other":
                duplicates.add(item.get())
            else:
                seen.add(item.get())
        if duplicates:
            messagebox.showwarning("Duplicate Warning", f"Resolve duplicte labels: {', '.join(duplicates)}")
            return

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
    btn_stop = tk.Button(button_frame, bg="#e0a2a2", activebackground="#e0a2a2", text="Stop All (a)", font=("Arial", 14), command=on_stop)
    btn_skip = tk.Button(button_frame, bg="#fff7a8", activebackground="#fff7a8", text="Skip (s)", font=("Arial", 14), command=on_skip)
    btn_done = tk.Button(button_frame, bg="#a2e0b2", activebackground="#a2e0b2", text="Approve (d)", font=("Arial", 14), command=on_done)

    # grid buttons
    btn_stop.grid(row=0, column=1, padx=10)
    btn_skip.grid(row=0, column=2, padx=10)
    btn_done.grid(row=0, column=3, padx=10)

    # add hotkeys
    root.bind("<a>", lambda event: btn_stop.invoke())
    root.bind("<s>", lambda event: btn_skip.invoke())
    root.bind("<d>", lambda event: btn_done.invoke())

    root.update()
