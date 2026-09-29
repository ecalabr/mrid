"""
Config wizard: a multi-page dialog that walks the user through generating or
editing a project config.py.

Pages:
  1. Output naming        (STUDY_NAME_TEMPLATE, thumbnail format/quality)
  2. Classification       (TARGET_SERIES / PREFER_3D, DICOM ImageType filters)
  3. Classification UI    (grid layout, thresholds, ordering)
  Advanced (optional)     (model hyperparameters, DICOM header fields, logging)

On save, the wizard rewrites a base config source (the project's existing
config.py, or the repository default for new projects) with the chosen
values, preserving all comments and formatting, and writes it to *save_path*.
"""

import os
import tkinter as tk

import customtkinter as ctk

from . import config as config_module
from . import ui_theme
from . import themed_dialogs as messagebox
from .config_io import (format_prefer3d_dict, rewrite_config_assignments,
                        get_default_config_path)
from .fusion import FUSION_METHODS, FUSION_METHOD_DESCRIPTIONS
from .logging_config import get_logger
from .ui_theme import scale_value, cascade

logger = get_logger()


class ConfigWizard:
    """Modal config wizard. Blocks until closed; .saved_path holds the result
    (None if cancelled)."""

    def __init__(self, master, base_config_path=None, save_path=None, modal=True,
                 analyze_mode=False):
        """
        Args:
            master: parent window
            base_config_path: config file used as the template and source of
                initial values (defaults to the repository config.py)
            save_path: where to write the generated config.py
            modal: block until the wizard is closed (False is used by tests)
            analyze_mode: lock the target-series section (Analyze mode will
                determine TARGET_SERIES/PREFER_3D from its clustering results)
        """
        self.master = master
        self.base_config_path = base_config_path or get_default_config_path()
        self.save_path = save_path
        self.saved_path = None
        self.analyze_mode = bool(analyze_mode)

        self._initial = self._load_initial_values()
        self._build_window()
        self._show_page(0)

        if modal:
            self.win.wait_window()

    # ── Initial values ────────────────────────────────────────────────────────

    def _load_initial_values(self):
        """Read initial values from the base config file (falling back to the
        in-memory config module for anything it lacks)."""
        values = {}
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("wizard_base_config", self.base_config_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception as e:
            logger.warning(f"Config wizard: could not read {self.base_config_path}: {e}")
            mod = None

        def get(name):
            if mod is not None and hasattr(mod, name):
                return getattr(mod, name)
            return getattr(config_module, name, None)

        for name in ["STUDY_NAME_TEMPLATE", "THUMBNAIL_FORMAT", "THUMBNAIL_QUALITY",
                     "TARGET_SERIES", "PREFER_3D", "INCLUDE_TYPES", "EXCLUDE_TYPES",
                     "IGNORE_MULTIPLE_ORIENTATIONS", "SEPARATE_STACKED_SERIES", "MAX_SPLIT",
                     "NUM_COLUMNS", "NUM_ROWS", "THUMBNAIL_SCALE_FACTOR", "OVERRIDE_THRESHOLD",
                     "ALLOW_DUPLICATE_LABELS", "NUM_PARALLEL_WORKERS", "RANDOMIZE_STUDY_ORDER",
                     "STUDY_ORDER_SEED", "FUSION_METHOD",
                     "IMAGE_SIZE", "NUM_EPOCHS", "BATCH_SIZE", "LR",
                     "CNN_NUM_SLICES", "N_ESTIMATORS", "DICOM_HEADER_FIELDS",
                     "ANALYZE_COVERAGE_THRESHOLD", "ANALYZE_HDBSCAN_CLUSTER_METHOD",
                     "SEMANTIC_MODEL_NAME", "LOG_LEVEL"]:
            values[name] = get(name)
        return values

    # ── Window scaffolding ────────────────────────────────────────────────────

    def _build_window(self):
        self.win = ctk.CTkToplevel(self.master)
        self.win.title("MRID Config Wizard")
        cascade(self.win, self.master)
        self.win.grab_set()
        self.win.protocol("WM_DELETE_WINDOW", self._on_cancel)

        self.header = ctk.CTkLabel(self.win, text="", font=ctk.CTkFont(size=16, weight="bold"))
        self.header.pack(pady=(scale_value(14), scale_value(4)))
        self.subheader = ctk.CTkLabel(self.win, text="", wraplength=scale_value(680),
                                      justify="left", text_color="#a0a0a0")
        self.subheader.pack(pady=(0, scale_value(6)))

        self.page_container = ctk.CTkFrame(self.win, fg_color="transparent")
        self.page_container.pack(fill="both", expand=True, padx=scale_value(16))

        nav = ctk.CTkFrame(self.win, fg_color="transparent")
        nav.pack(fill="x", pady=scale_value(12), padx=scale_value(16))
        self.back_btn = ctk.CTkButton(nav, text=ui_theme.t("← Back"), width=100,
                                      fg_color=ui_theme.NEUTRAL, hover_color=ui_theme.NEUTRAL_HOVER,
                                      command=self._on_back)
        self.back_btn.pack(side="left")
        self.cancel_btn = ctk.CTkButton(nav, text="Cancel", width=100,
                                        fg_color=ui_theme.NEUTRAL, hover_color=ui_theme.NEUTRAL_HOVER,
                                        command=self._on_cancel)
        self.cancel_btn.pack(side="left", padx=scale_value(8))
        self.next_btn = ctk.CTkButton(nav, text=ui_theme.t("Next →"), width=120, command=self._on_next)
        self.next_btn.pack(side="right")
        self.advanced_btn = ctk.CTkButton(nav, text=ui_theme.t("Advanced Settings…"), width=160,
                                          fg_color=ui_theme.YELLOW, hover_color=ui_theme.YELLOW_HOVER,
                                          command=self._on_advanced)
        # advanced_btn is only packed on the last standard page

        self.pages = [self._build_page_naming(),
                      self._build_page_classification(),
                      self._build_page_ui()]
        self.advanced_page = self._build_page_advanced()
        self.current = 0
        self.on_advanced_page = False

    def _show_page(self, index, advanced=False):
        for page in self.pages + [self.advanced_page]:
            page["frame"].pack_forget()
        self.advanced_btn.pack_forget()

        if advanced:
            page = self.advanced_page
            self.on_advanced_page = True
            self.next_btn.configure(text="Save Config",
                                    fg_color=ui_theme.GREEN, hover_color=ui_theme.GREEN_HOVER)
            self.back_btn.configure(state="normal")
        else:
            self.current = index
            self.on_advanced_page = False
            page = self.pages[index]
            last = index == len(self.pages) - 1
            if last:
                self.next_btn.configure(text="Save Config",
                                        fg_color=ui_theme.GREEN, hover_color=ui_theme.GREEN_HOVER)
                self.advanced_btn.pack(side="right", padx=scale_value(8))
            else:
                self.next_btn.configure(text=ui_theme.t("Next →"),
                                        fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"],
                                        hover_color=ctk.ThemeManager.theme["CTkButton"]["hover_color"])
            self.back_btn.configure(state="normal" if index > 0 else "disabled")

        self.header.configure(text=page["title"])
        self.subheader.configure(text=page["subtitle"])
        page["frame"].pack(fill="both", expand=True)

        # Size the window to the current page so nothing is cut off
        # (real measured pixels, so bypass CTk geometry scaling)
        self.win.update_idletasks()
        width = max(self.win.winfo_reqwidth(), scale_value(760))
        height = self.win.winfo_reqheight()
        self.win.wm_geometry(f"{width}x{height}")

    def _on_back(self):
        if self.on_advanced_page:
            self._show_page(len(self.pages) - 1)
        elif self.current > 0:
            self._show_page(self.current - 1)

    def _on_next(self):
        if self.on_advanced_page or self.current == len(self.pages) - 1:
            self._save()
        else:
            self._show_page(self.current + 1)

    def _on_advanced(self):
        self._show_page(0, advanced=True)

    def _on_cancel(self):
        self.saved_path = None
        self.win.grab_release()
        self.win.destroy()

    # ── Page builders ─────────────────────────────────────────────────────────

    def _entry_row(self, parent, row, label, initial, width=220, hint=None):
        ctk.CTkLabel(parent, text=label, anchor="w").grid(
            row=row, column=0, sticky="w", padx=scale_value(6), pady=scale_value(5))
        var = tk.StringVar(value="" if initial is None else str(initial))
        ctk.CTkEntry(parent, textvariable=var, width=width).grid(
            row=row, column=1, sticky="w", padx=scale_value(6), pady=scale_value(5))
        if hint:
            ctk.CTkLabel(parent, text=hint, text_color="#8a8a8a", anchor="w").grid(
                row=row, column=2, sticky="w", padx=scale_value(6))
        return var

    def _option_row(self, parent, row, label, initial, values, width=140,
                    hint=None, tooltips=None):
        """A labelled dropdown row.

        *tooltips* maps an option value to a longer description; when supplied
        the dropdown carries a hover tooltip that follows the current selection,
        so each choice can explain itself without crowding the page.
        """
        ctk.CTkLabel(parent, text=label, anchor="w").grid(
            row=row, column=0, sticky="w", padx=scale_value(6), pady=scale_value(5))
        var = tk.StringVar(value=str(initial))
        menu = ctk.CTkOptionMenu(parent, variable=var, width=width, values=list(values))
        menu.grid(row=row, column=1, sticky="w", padx=scale_value(6), pady=scale_value(5))
        if hint:
            ctk.CTkLabel(parent, text=hint, text_color="#8a8a8a", anchor="w").grid(
                row=row, column=2, sticky="w", padx=scale_value(6))
        if tooltips:
            tip = ui_theme.ToolTip(menu, tooltips.get(var.get(), ""))
            var.trace_add("write", lambda *_a: tip.update_text(tooltips.get(var.get(), "")))
        return var

    def _check_row(self, parent, row, label, initial, hint=None):
        var = tk.BooleanVar(value=bool(initial))
        ctk.CTkCheckBox(parent, text=label, variable=var).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=scale_value(6), pady=scale_value(5))
        if hint:
            ctk.CTkLabel(parent, text=hint, text_color="#8a8a8a", anchor="w").grid(
                row=row, column=2, sticky="w", padx=scale_value(6))
        return var

    def _build_page_naming(self):
        frame = ctk.CTkFrame(self.page_container, fg_color="transparent")
        grid = ctk.CTkFrame(frame, fg_color="transparent")
        grid.pack(anchor="nw", pady=scale_value(8))

        self.v_template = self._entry_row(
            grid, 0, "Study name template:", self._initial["STUDY_NAME_TEMPLATE"],
            hint="Placeholders: {mrn}, {accn}, {study_uid}")
        ctk.CTkLabel(grid, text="Thumbnail format:", anchor="w").grid(
            row=1, column=0, sticky="w", padx=scale_value(6), pady=scale_value(5))
        self.v_thumb_format = tk.StringVar(value=self._initial["THUMBNAIL_FORMAT"] or "JPEG")
        ctk.CTkOptionMenu(grid, variable=self.v_thumb_format, values=["JPEG", "PNG"],
                          width=120).grid(row=1, column=1, sticky="w", padx=scale_value(6))
        self.v_thumb_quality = self._entry_row(
            grid, 2, "Thumbnail JPEG quality:", self._initial["THUMBNAIL_QUALITY"],
            width=80, hint="1–100 (JPEG only)")

        return {"frame": frame, "title": "Step 1 of 3 — Output Naming",
                "subtitle": "How classified study folders and thumbnails are named and stored."}

    def _build_page_classification(self):
        frame = ctk.CTkFrame(self.page_container, fg_color="transparent")

        if self.analyze_mode:
            ctk.CTkLabel(frame, text="Target series (locked):",
                         anchor="w").pack(anchor="w", padx=scale_value(6))
            ctk.CTkLabel(frame,
                         text="Target series will be determined from the results of Analyze "
                              "mode — name the discovered clusters when the analysis completes.",
                         text_color="#9c8a1c", wraplength=scale_value(660),
                         justify="left", anchor="w").pack(anchor="w", padx=scale_value(6))
        else:
            ctk.CTkLabel(frame, text="Target series (the labels you will classify into; "
                                     "\"Other\" is always added automatically):",
                         anchor="w").pack(anchor="w", padx=scale_value(6))

        self.series_rows_frame = ctk.CTkScrollableFrame(frame, height=190)
        self.series_rows_frame.pack(fill="x", padx=scale_value(6), pady=scale_value(6))
        self.series_rows = []  # list of (name_var, prefer3d_var, row_frame)

        prefer3d = self._initial["PREFER_3D"]
        for name in (self._initial["TARGET_SERIES"] or []):
            if str(name).strip().lower() == "other":
                continue
            if isinstance(prefer3d, dict):
                p3d = bool(prefer3d.get(name, False))
            else:
                p3d = bool(prefer3d)
            self._add_series_row(name, p3d)

        add_btn = ctk.CTkButton(frame, text="+ Add series", width=110,
                                fg_color=ui_theme.NEUTRAL, hover_color=ui_theme.NEUTRAL_HOVER,
                                command=lambda: self._add_series_row("", False))
        add_btn.pack(anchor="w", padx=scale_value(6), pady=(0, scale_value(8)))

        if self.analyze_mode:
            add_btn.configure(state="disabled", fg_color=ui_theme.DISABLED)

        grid = ctk.CTkFrame(frame, fg_color="transparent")
        grid.pack(anchor="nw", pady=scale_value(4))
        inc = self._initial["INCLUDE_TYPES"] or []
        exc = self._initial["EXCLUDE_TYPES"] or []
        self.v_include = self._entry_row(grid, 0, "ImageType must include:", ", ".join(inc),
                                         width=320, hint="comma-separated")
        self.v_exclude = self._entry_row(grid, 1, "ImageType must NOT include:", ", ".join(exc),
                                         width=320, hint="comma-separated")
        self.v_ignore_multi = self._check_row(grid, 2, "Ignore series with multiple orientations",
                                              self._initial["IGNORE_MULTIPLE_ORIENTATIONS"])
        self.v_separate = self._check_row(grid, 3, "Split stacked series",
                                          self._initial["SEPARATE_STACKED_SERIES"])
        self.v_max_split = self._entry_row(grid, 4, "Max splits per series:",
                                           self._initial["MAX_SPLIT"], width=80)

        return {"frame": frame, "title": "Step 2 of 3 — Classification & DICOM Filtering",
                "subtitle": "Which series types to classify and which DICOM files to consider."}

    def _add_series_row(self, name, prefer3d):
        row = ctk.CTkFrame(self.series_rows_frame, fg_color="transparent")
        row.pack(fill="x", pady=2)
        name_var = tk.StringVar(value=str(name))
        p3d_var = tk.BooleanVar(value=bool(prefer3d))
        name_entry = ctk.CTkEntry(row, textvariable=name_var, width=200)
        name_entry.pack(side="left", padx=scale_value(4))
        p3d_cb = ctk.CTkCheckBox(row, text="Prefer 3D acquisition", variable=p3d_var)
        p3d_cb.pack(side="left", padx=scale_value(10))
        entry = (name_var, p3d_var, row)

        def remove():
            try:
                self.series_rows.remove(entry)
            except ValueError:
                pass
            row.destroy()

        remove_btn = ctk.CTkButton(row, text=ui_theme.t("✕"), width=30, fg_color=ui_theme.RED,
                                   hover_color=ui_theme.RED_HOVER, command=remove)
        remove_btn.pack(side="left")
        if self.analyze_mode:
            name_entry.configure(state="disabled")
            p3d_cb.configure(state="disabled")
            remove_btn.configure(state="disabled", fg_color=ui_theme.DISABLED)
        self.series_rows.append(entry)

    def _build_page_ui(self):
        frame = ctk.CTkFrame(self.page_container, fg_color="transparent")
        grid = ctk.CTkFrame(frame, fg_color="transparent")
        grid.pack(anchor="nw", pady=scale_value(8))

        self.v_columns = self._entry_row(grid, 0, "Grid columns:", self._initial["NUM_COLUMNS"], width=80)
        self.v_rows = self._entry_row(grid, 1, "Grid rows (Verify mode):", self._initial["NUM_ROWS"], width=80)
        self.v_thumb_scale = self._entry_row(grid, 2, "Thumbnail scale factor:",
                                             self._initial["THUMBNAIL_SCALE_FACTOR"], width=80,
                                             hint="e.g. 0.75 to shrink, 1.5 to enlarge")
        self.v_threshold = self._entry_row(grid, 3, "Auto-classification threshold:",
                                           self._initial["OVERRIDE_THRESHOLD"], width=80,
                                           hint="probability 0–1 for Semi/Full-Auto")
        self.v_fusion = self._option_row(
            grid, 4, "Header + Image fusion:",
            self._initial["FUSION_METHOD"] or "reliability",
            list(FUSION_METHODS), width=140,
            hint="how the two models' probabilities are combined",
            tooltips=FUSION_METHOD_DESCRIPTIONS)
        self.v_workers = self._entry_row(grid, 5, "Parallel workers:",
                                         self._initial["NUM_PARALLEL_WORKERS"], width=80)
        self.v_dup_labels = self._check_row(grid, 6, "Allow duplicate labels within a study",
                                            self._initial["ALLOW_DUPLICATE_LABELS"])
        self.v_randomize = self._check_row(grid, 7, "Randomize study order (Manual/Semi-Auto)",
                                           self._initial["RANDOMIZE_STUDY_ORDER"])
        self.v_seed = self._entry_row(grid, 8, "Study order seed:",
                                      self._initial["STUDY_ORDER_SEED"], width=80)

        return {"frame": frame, "title": "Step 3 of 3 — Classification UI",
                "subtitle": "Layout and behavior of the classification and verification windows."}

    def _build_page_advanced(self):
        frame = ctk.CTkFrame(self.page_container, fg_color="transparent")
        grid = ctk.CTkFrame(frame, fg_color="transparent")
        grid.pack(anchor="nw", pady=scale_value(4))

        self.v_image_size = self._entry_row(grid, 0, "CNN image size:", self._initial["IMAGE_SIZE"], width=80)
        self.v_epochs = self._entry_row(grid, 1, "CNN epochs:", self._initial["NUM_EPOCHS"], width=80)
        self.v_batch = self._entry_row(grid, 2, "CNN batch size:", self._initial["BATCH_SIZE"], width=80)
        self.v_lr = self._entry_row(grid, 3, "CNN learning rate:", self._initial["LR"], width=80)
        self.v_cnn_slices = self._entry_row(grid, 4, "CNN slices (2.5D):", self._initial["CNN_NUM_SLICES"],
                                            width=80, hint="1 = center slice only")
        self.v_estimators = self._entry_row(grid, 5, "Random Forest trees:", self._initial["N_ESTIMATORS"], width=80)
        self.v_semantic_model = self._entry_row(
            grid, 9, "Text embedding model:", self._initial["SEMANTIC_MODEL_NAME"], width=220,
            hint="sentence-transformers model (retrain header model after changing)")
        self.v_coverage = self._entry_row(grid, 6, "Analyze coverage threshold:",
                                          self._initial["ANALYZE_COVERAGE_THRESHOLD"], width=80)
        ctk.CTkLabel(grid, text="Analyze HDBSCAN method:", anchor="w").grid(
            row=7, column=0, sticky="w", padx=scale_value(6), pady=scale_value(5))
        self.v_hdbscan_method = tk.StringVar(
            value=(self._initial["ANALYZE_HDBSCAN_CLUSTER_METHOD"] or "leaf"))
        ctk.CTkOptionMenu(grid, variable=self.v_hdbscan_method, width=120,
                          values=["leaf", "eom"]).grid(
            row=7, column=1, sticky="w", padx=scale_value(6))
        ctk.CTkLabel(grid, text="leaf = more/finer clusters, eom = fewer/larger",
                     text_color="#8a8a8a", anchor="w").grid(
            row=7, column=2, sticky="w", padx=scale_value(6))
        ctk.CTkLabel(grid, text="Log level:", anchor="w").grid(
            row=8, column=0, sticky="w", padx=scale_value(6), pady=scale_value(5))
        self.v_log_level = tk.StringVar(value=self._initial["LOG_LEVEL"] or "INFO")
        ctk.CTkOptionMenu(grid, variable=self.v_log_level, width=120,
                          values=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]).grid(
            row=8, column=1, sticky="w", padx=scale_value(6))

        ctk.CTkLabel(frame, text="DICOM header fields for the header model "
                                 "(one per line: Display Name = DICOMAttribute):",
                     anchor="w").pack(anchor="w", padx=scale_value(6), pady=(scale_value(8), 0))
        self.fields_box = ctk.CTkTextbox(frame, height=170)
        self.fields_box.pack(fill="both", expand=True, padx=scale_value(6), pady=scale_value(6))
        for display, attr in (self._initial["DICOM_HEADER_FIELDS"] or []):
            self.fields_box.insert("end", f"{display} = {attr}\n")

        return {"frame": frame, "title": "Advanced Settings",
                "subtitle": "Model hyperparameters and DICOM header fields. "
                            "Most projects can keep the defaults."}

    # ── Validation & save ─────────────────────────────────────────────────────

    def _parse(self, var, kind, name, lo=None, hi=None):
        raw = var.get().strip()
        try:
            value = kind(raw)
        except Exception:
            raise ValueError(f"{name} must be a {kind.__name__} (got {raw!r})")
        if lo is not None and value < lo:
            raise ValueError(f"{name} must be ≥ {lo}")
        if hi is not None and value > hi:
            raise ValueError(f"{name} must be ≤ {hi}")
        return value

    def _collect(self):
        """Validate all inputs and return a {VAR: repr_string} dict."""
        template = self.v_template.get().strip()
        if not template:
            raise ValueError("Study name template cannot be empty")

        if self.analyze_mode:
            # Target series are locked: Analyze mode writes them from its
            # clustering results, so keep the base config's values
            names = list(self._initial["TARGET_SERIES"] or ["Other"])
            initial_p3d = self._initial["PREFER_3D"]
            if isinstance(initial_p3d, dict):
                prefer3d = {n: bool(initial_p3d.get(n, False)) for n in names}
            else:
                prefer3d = {n: (bool(initial_p3d) and n.lower() != "other") for n in names}
        else:
            names, prefer3d = [], {}
            for name_var, p3d_var, _row in self.series_rows:
                name = name_var.get().strip()
                if not name:
                    continue
                if name.lower() == "other":
                    continue
                if name in names:
                    raise ValueError(f"Duplicate series name: {name}")
                names.append(name)
                prefer3d[name] = bool(p3d_var.get())
            if not names:
                raise ValueError("At least one target series is required")
            names.append("Other")
            prefer3d["Other"] = False

        include = [s.strip() for s in self.v_include.get().split(",") if s.strip()]
        exclude = [s.strip() for s in self.v_exclude.get().split(",") if s.strip()]

        semantic_model = self.v_semantic_model.get().strip()
        if not semantic_model:
            raise ValueError("Text embedding model cannot be empty")

        fields = []
        for line in self.fields_box.get("1.0", "end").splitlines():
            line = line.strip()
            if not line:
                continue
            if "=" not in line:
                raise ValueError(f"DICOM header field line must be 'Display Name = Attribute': {line!r}")
            display, attr = (part.strip() for part in line.split("=", 1))
            if not display or not attr:
                raise ValueError(f"Invalid DICOM header field line: {line!r}")
            fields.append((display, attr))
        if not fields:
            raise ValueError("At least one DICOM header field is required")

        return {
            "STUDY_NAME_TEMPLATE": repr(template),
            "THUMBNAIL_FORMAT": repr(self.v_thumb_format.get()),
            "THUMBNAIL_QUALITY": repr(self._parse(self.v_thumb_quality, int, "Thumbnail quality", 1, 100)),
            "TARGET_SERIES": repr(names),
            "PREFER_3D": format_prefer3d_dict(prefer3d),
            "INCLUDE_TYPES": repr(include),
            "EXCLUDE_TYPES": repr(exclude),
            "IGNORE_MULTIPLE_ORIENTATIONS": repr(bool(self.v_ignore_multi.get())),
            "SEPARATE_STACKED_SERIES": repr(bool(self.v_separate.get())),
            "MAX_SPLIT": repr(self._parse(self.v_max_split, int, "Max splits", 1)),
            "NUM_COLUMNS": repr(self._parse(self.v_columns, int, "Grid columns", 1)),
            "NUM_ROWS": repr(self._parse(self.v_rows, int, "Grid rows", 1)),
            "THUMBNAIL_SCALE_FACTOR": repr(self._parse(self.v_thumb_scale, float, "Thumbnail scale", 0.1)),
            "OVERRIDE_THRESHOLD": repr(self._parse(self.v_threshold, float, "Threshold", 0.0, 1.0)),
            "ALLOW_DUPLICATE_LABELS": repr(bool(self.v_dup_labels.get())),
            "NUM_PARALLEL_WORKERS": repr(self._parse(self.v_workers, int, "Parallel workers", 1)),
            "RANDOMIZE_STUDY_ORDER": repr(bool(self.v_randomize.get())),
            "STUDY_ORDER_SEED": repr(self._parse(self.v_seed, int, "Study order seed")),
            "FUSION_METHOD": repr(self.v_fusion.get()),
            "IMAGE_SIZE": repr(self._parse(self.v_image_size, int, "CNN image size", 16)),
            "NUM_EPOCHS": repr(self._parse(self.v_epochs, int, "CNN epochs", 1)),
            "BATCH_SIZE": repr(self._parse(self.v_batch, int, "CNN batch size", 1)),
            "LR": repr(self._parse(self.v_lr, float, "Learning rate", 0.0)),
            "CNN_NUM_SLICES": repr(self._parse(self.v_cnn_slices, int, "CNN slices", 1)),
            "N_ESTIMATORS": repr(self._parse(self.v_estimators, int, "Random Forest trees", 1)),
            "ANALYZE_COVERAGE_THRESHOLD": repr(self._parse(self.v_coverage, float, "Coverage threshold", 0.0, 1.0)),
            "ANALYZE_HDBSCAN_CLUSTER_METHOD": repr(self.v_hdbscan_method.get()),
            "SEMANTIC_MODEL_NAME": repr(semantic_model),
            "LOG_LEVEL": repr(self.v_log_level.get()),
            "DICOM_HEADER_FIELDS": "[\n" + "".join(
                f"    ({display!r}, {attr!r}),\n" for display, attr in fields) + "]",
        }

    def _save(self):
        try:
            replacements = self._collect()
        except ValueError as e:
            messagebox.showwarning("Invalid Setting", str(e), parent=self.win)
            return

        if not self.save_path:
            messagebox.showerror("No Save Location", "No save path was provided for the config file.",
                                 parent=self.win)
            return

        try:
            with open(self.base_config_path, "r", encoding="utf-8") as fh:
                base_source = fh.read()
            new_source = rewrite_config_assignments(base_source, replacements)
            os.makedirs(os.path.dirname(self.save_path) or ".", exist_ok=True)
            with open(self.save_path, "w", encoding="utf-8") as fh:
                fh.write(new_source)
        except Exception as e:
            messagebox.showerror("Save Failed", f"Could not write config file:\n{e}", parent=self.win)
            return

        logger.info(f"Config wizard wrote: {self.save_path}")
        self.saved_path = self.save_path
        self.win.grab_release()
        self.win.destroy()


def run_config_wizard(master, base_config_path=None, save_path=None, analyze_mode=False):
    """Run the wizard modally. Returns the saved config path, or None."""
    wizard = ConfigWizard(master, base_config_path=base_config_path, save_path=save_path,
                          analyze_mode=analyze_mode)
    return wizard.saved_path
