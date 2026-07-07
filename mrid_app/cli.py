"""Console entry point for the ``mrid`` command.

Installed via the ``[project.scripts]`` table in ``pyproject.toml``:

    mrid                 Launch the GUI (same as ``python run.py``)
    mrid auto [options]  Run headless full-auto classification
                         (same as ``python headless_autoclassify.py [options]``)

The underlying ``run.py`` and ``headless_autoclassify.py`` scripts remain usable
directly; this just provides a single installed command.
"""
import sys

_USAGE = """\
mrid — MRI DICOM classification tool

Usage:
  mrid                  Launch the GUI
  mrid auto [options]   Run headless full-auto classification
  mrid auto --help      List all headless options

Examples:
  mrid
  mrid auto -i /data/dicoms -o /data/output --model-type combined
"""


def main(argv=None):
    """Dispatch to the GUI or the headless CLI based on the first argument."""
    args = list(sys.argv[1:] if argv is None else argv)

    if args and args[0] == "auto":
        # Hand the remaining arguments to the headless CLI. Rewrite argv[0] so
        # its argparse usage/help reads "mrid auto ...".
        from headless_autoclassify import main as auto_main
        sys.argv = ["mrid auto", *args[1:]]
        auto_main()
    elif args and args[0] in ("help", "-h", "--help"):
        print(_USAGE)
    else:
        # No subcommand launches the GUI (any extra args are ignored by the GUI).
        from run import main as gui_main
        gui_main()


if __name__ == "__main__":
    main()
