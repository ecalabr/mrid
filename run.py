"""Entry point to launch the MRID GUI application."""
import customtkinter as ctk
from mrid_app import ui_theme
from mrid_app.gui import MainApp


def main():
    ui_theme.init_appearance()
    root = ctk.CTk()
    ui_theme.init_scaling(root)
    app = MainApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
