"""mrid_app package

This package contains modules split out from the original single-file
`mrid.py` script for better organization: config, cnn, header model,
preprocessing, GUI, and utilities.

Import the application entrypoint with:

    from mrid_app.run import main

or run `python run.py` from the project root.
"""

# Ensure the HuggingFace tokenizers parallelism env var is set before any
# tokenizers/transformers code runs. This prevents the runtime message when
# the process is forked after tokenizers has used parallelism.
# We default to 'false' (recommended) to avoid deadlocks when multiprocessing
# is used later in the application.
try:
    import os
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
except Exception:
    pass

# Suppress noisy pydicom valuerep warnings that appear on stdout/terminal.
# These warnings are benign for our use-case and clutter output during scans.
try:
    import warnings
    # Ignore UserWarning coming from pydicom.valrep (valuerep module)
    warnings.filterwarnings("ignore", category=UserWarning, module=r"pydicom\.valrep")
    # Also ignore any pydicom-related warnings more broadly
    warnings.filterwarnings("ignore", category=UserWarning, module=r"pydicom(\.|$)")
    # Additionally, wrap the global showwarning to suppress warnings that originate
    # from files in the pydicom package (filename contains 'pydicom'). This is a
    # robust fallback for environments where the module-based filter doesn't match.
    try:
        _orig_showwarning = warnings.showwarning

        def _showwarning_wrapper(message, category, filename, lineno, file=None, line=None):
            try:
                # If the provided filename indicates pydicom, suppress immediately
                if filename and 'pydicom' in str(filename):
                    return
                # Otherwise inspect the call stack to determine whether the warning
                # originated from inside the pydicom package (file paths containing 'pydicom').
                try:
                    import traceback
                    for frame in traceback.extract_stack():
                        if frame and frame.filename and 'pydicom' in frame.filename:
                            # suppress warnings originating from pydicom internals
                            return
                except Exception:
                    pass
            except Exception:
                pass
            return _orig_showwarning(message, category, filename, lineno, file=file, line=line)

        warnings.showwarning = _showwarning_wrapper
    except Exception:
        pass
except Exception:
    # If warnings or filterrules fail for any reason, continue without crashing.
    pass

__all__ = [
    "config",
    "cnn",
    "header",
    "prep",
    "gui",
    "utils",
]
