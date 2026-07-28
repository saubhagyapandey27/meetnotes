import tkinter as tk
from tkinter import ttk

# Modern Dark Theme Constants
BG_COLOR = "#0D0E12"        # Slate Black
CARD_BG = "#161822"         # Deep Slate Gray
ACCENT_COLOR = "#6366F1"    # Indigo Accent (Indigo 500)
ACCENT_HOVER = "#4F46E5"    # Indigo 600
TEXT_MAIN = "#F3F4F6"       # Light gray (Gray 100)
TEXT_MUTED = "#9CA3AF"      # Gray 400
TEXT_SUCCESS = "#10B981"    # Emerald Green (Green 500)
BORDER_COLOR = "#2A2D3D"    # Slate Border
TEXT_FONT = ("Inter", 11)
TITLE_FONT = ("Outfit", 18, "bold")
MONOSPACE_FONT = ("Consolas", 10)

def apply_dark_theme(root: tk.Tk):
    """
    Applies custom styles to standard and ttk Tkinter widgets.
    """
    # Set main window background
    root.configure(bg=BG_COLOR)
    
    # Configure ttk style manager
    style = ttk.Style()
    style.theme_use("clam")
    
    # Configure Notebook (Tabs)
    style.configure(
        "TNotebook",
        background=BG_COLOR,
        borderwidth=0
    )
    style.configure(
        "TNotebook.Tab",
        background=CARD_BG,
        foreground=TEXT_MUTED,
        font=("Inter", 10, "bold"),
        padding=[15, 6],
        borderwidth=0
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", ACCENT_COLOR)],
        foreground=[("selected", TEXT_MAIN)]
    )
    
    # Configure Frames
    style.configure(
        "TFrame",
        background=BG_COLOR,
        borderwidth=0
    )
    style.configure(
        "Card.TFrame",
        background=CARD_BG,
        borderwidth=1,
        relief="solid",
        bordercolor=BORDER_COLOR
    )
    
    # Configure Labels
    style.configure(
        "TLabel",
        background=BG_COLOR,
        foreground=TEXT_MAIN,
        font=TEXT_FONT
    )
    style.configure(
        "Muted.TLabel",
        background=BG_COLOR,
        foreground=TEXT_MUTED,
        font=TEXT_FONT
    )
    style.configure(
        "Card.TLabel",
        background=CARD_BG,
        foreground=TEXT_MAIN,
        font=TEXT_FONT
    )
    style.configure(
        "CardTitle.TLabel",
        background=CARD_BG,
        foreground=TEXT_MAIN,
        font=("Outfit", 14, "bold")
    )
    
    # Configure Buttons
    style.configure(
        "TButton",
        background=ACCENT_COLOR,
        foreground=TEXT_MAIN,
        font=("Inter", 10, "bold"),
        padding=[12, 6],
        borderwidth=0
    )
    style.map(
        "TButton",
        background=[("active", ACCENT_HOVER)],
        foreground=[("active", TEXT_MAIN)]
    )
    
    style.configure(
        "Muted.TButton",
        background=CARD_BG,
        foreground=TEXT_MUTED,
        font=("Inter", 10, "bold"),
        padding=[12, 6],
        borderwidth=1,
        relief="solid",
        bordercolor=BORDER_COLOR
    )
    style.map(
        "Muted.TButton",
        background=[("active", BORDER_COLOR)],
        foreground=[("active", TEXT_MAIN)]
    )
    
    # Configure Checkbuttons
    style.configure(
        "TCheckbutton",
        background=BG_COLOR,
        foreground=TEXT_MAIN,
        font=TEXT_FONT
    )
    style.map(
        "TCheckbutton",
        background=[("active", BG_COLOR)],
        foreground=[("active", TEXT_MAIN)]
    )
    
    # Configure Treeview (past recordings table)
    style.configure(
        "Treeview",
        background=CARD_BG,
        foreground=TEXT_MAIN,
        fieldbackground=CARD_BG,
        font=TEXT_FONT,
        rowheight=28,
        borderwidth=0
    )
    style.configure(
        "Treeview.Heading",
        background=BORDER_COLOR,
        foreground=TEXT_MAIN,
        font=("Inter", 10, "bold"),
        borderwidth=0
    )
    style.map(
        "Treeview",
        background=[("selected", ACCENT_COLOR)],
        foreground=[("selected", TEXT_MAIN)]
    )
    
    # Configure Progressbar
    style.configure(
        "TProgressbar",
        troughcolor=CARD_BG,
        background=ACCENT_COLOR,
        thickness=8,
        borderwidth=0
    )
