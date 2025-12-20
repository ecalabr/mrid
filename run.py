"""Entry point to launch the MRID GUI application."""
import tkinter as tk
from mrid_app.gui import MainApp


def main():
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
