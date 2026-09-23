import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd

# Enable high-DPI scaling on Windows to keep fonts and widgets sharp
try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

from crop_yield_prediction import (
    get_dataset,
    get_trained_crops,
    clean_name,
    predict_crop,
    compare_models,
    detailed_prediction
)
import detailed_view

# Brand Color Palette (Refined Agri-Tech Theme)
PRIMARY = "#1B4D2E"       # Deep evergreen
PRIMARY_HOVER = "#143A23"
SECONDARY = "#2E7D32"     # Leaf green
ACCENT_BLUE = "#1565C0"   # Metric accent
ACCENT_AMBER = "#E65100"  # Notice accent
BG_COLOR = "#F4F7F4"      # Soft light background
CARD_BG = "#FFFFFF"       # Clean card white
CARD_BORDER = "#D8E2D8"   # Subtle card border
TEXT_DARK = "#1C251D"     # High-contrast readable text
TEXT_MUTED = "#5C6B5E"    # Secondary labels


class CropYieldApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Crop Yield Prediction System")
        self.root.geometry("1240x820")
        self.root.minsize(1060, 700)
        self.root.configure(bg=BG_COLOR)

        self._center_window()
        self._load_data()
        self._init_variables()
        self._setup_styles()
        self._build_ui()
        self._bind_shortcuts()

    def _center_window(self):
        self.root.update_idletasks()
        w = 1240
        h = 820
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x = max(0, int((sw - w) / 2))
        y = max(0, int((sh - h) / 2) - 20)
        self.root.geometry(f"{w}x{h}+{x}+{y}")

    def _load_data(self):
        """Load dataset and restrict choices to crops with verified trained models."""
        raw_df = get_dataset()

        # Only retain records with positive historical yield
        valid_df = raw_df[raw_df["yield"] > 0].copy()

        # Restrict selectable crops strictly to models present in models/rf and models/linear
        trained_slugs = get_trained_crops()
        crop_mask = valid_df["crop_name"].apply(lambda c: clean_name(c) in trained_slugs)
        self.df = valid_df[crop_mask].copy()

        # Uppercase representations for clean display and exact cascading
        self.df["STATE_UPPER"] = self.df["state_name"].astype(str).str.strip().str.upper()
        self.df["DISTRICT_UPPER"] = self.df["district_name"].astype(str).str.strip().str.upper()
        self.df["CROP_UPPER"] = self.df["crop_name"].astype(str).str.strip().str.upper()
        self.df["SEASON_UPPER"] = self.df["season"].astype(str).str.strip().str.upper()

        self.states = sorted(self.df["STATE_UPPER"].dropna().unique())
        self.available_crop_count = self.df["CROP_UPPER"].nunique()

    def _init_variables(self):
        self.state_var = tk.StringVar()
        self.district_var = tk.StringVar()
        self.crop_var = tk.StringVar()
        self.season_var = tk.StringVar()

        # Active model selection: 'rf' (Random Forest) or 'linear' (Ridge)
        self.model_type_var = tk.StringVar(value="rf")

        self.last_prediction_data = None

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        # Combobox styling
        style.configure(
            "TCombobox",
            padding=6,
            font=("Segoe UI", 10),
            fieldbackground="white",
            background="#E8EFE8"
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", "white")],
            selectbackground=[("readonly", "#C8E6C9")],
            selectforeground=[("readonly", TEXT_DARK)]
        )

        # Radiobutton styling
        style.configure(
            "Model.TRadiobutton",
            background=CARD_BG,
            foreground=TEXT_DARK,
            font=("Segoe UI", 10, "bold"),
            padding=4
        )

        # Progressbar styling
        style.configure(
            "Green.Horizontal.TProgressbar",
            troughcolor="#E8EFE8",
            background=SECONDARY,
            thickness=8
        )

        # Separator
        style.configure(
            "TSeparator",
            background=CARD_BORDER
        )

    def _build_ui(self):
        # 1. Header Banner
        header = tk.Frame(self.root, bg=PRIMARY)
        header.pack(fill="x")

        h_left = tk.Frame(header, bg=PRIMARY)
        h_left.pack(side="left", padx=24, pady=12)

        tk.Label(
            h_left,
            text="🌾 CROP YIELD PREDICTION SYSTEM",
            bg=PRIMARY,
            fg="#FFFFFF",
            font=("Segoe UI", 16, "bold")
        ).pack(anchor="w")

        tk.Label(
            h_left,
            text="Predict crop yield using Random Forest and Ridge Regression",
            bg=PRIMARY,
            fg="#D7ECD9",
            font=("Segoe UI", 10)
        ).pack(anchor="w", pady=(3, 0))

        # Dataset Stats Pills
        h_right = tk.Frame(header, bg=PRIMARY)
        h_right.pack(side="right", padx=20, pady=12)

        stats = [
            (f"{len(self.df):,}", "Total Records"),
            (f"{self.df['STATE_UPPER'].nunique()}", "States"),
            (f"{self.df['DISTRICT_UPPER'].nunique()}", "Districts"),
            (f"{self.available_crop_count}", "Crops"),
            ("2", "Models")
        ]

        for val, lbl in stats:
            chip = tk.Frame(h_right, bg="#205833", padx=10, pady=4, bd=1, relief="flat")
            chip.pack(side="left", padx=4)
            tk.Label(chip, text=val, bg="#205833", fg="#FFFFFF", font=("Segoe UI", 10, "bold")).pack()
            tk.Label(chip, text=lbl, bg="#205833", fg="#A5D6A7", font=("Segoe UI", 7)).pack()

        # Main Layout Container
        main_box = tk.Frame(self.root, bg=BG_COLOR, padx=22, pady=12)
        main_box.pack(fill="both", expand=True)

        # 2. Parameter Input Card
        self._build_parameter_card(main_box)

        # 3. Execution Progress & Loading Card
        self._build_progress_card(main_box)

        # 4. Model Results & Analytics Frame
        results_frame = tk.Frame(main_box, bg=BG_COLOR)
        results_frame.pack(fill="both", expand=True, pady=8)

        # Left: Yield Prediction Display Card
        self._build_prediction_card(results_frame)

        # Right: Model Performance & Accuracy Card
        self._build_performance_card(results_frame)

        # 5. Action Buttons Bar
        self._build_action_bar(main_box)

        # 6. Status Footer
        footer = tk.Frame(self.root, bg="#E8EFE8", height=28)
        footer.pack(fill="x", side="bottom")
        footer.pack_propagate(False)

        self.status_label = tk.Label(
            footer,
            text="Ready  •  Select parameters and click Predict Yield",
            bg="#E8EFE8",
            fg=TEXT_MUTED,
            font=("Segoe UI", 9)
        )
        self.status_label.pack(side="left", padx=20, pady=3)

        tk.Label(
            footer,
            text="Python 3.14  •  Scikit-Learn  •  Tkinter",
            bg="#E8EFE8",
            fg=TEXT_MUTED,
            font=("Segoe UI", 9)
        ).pack(side="right", padx=20, pady=3)

    def _build_parameter_card(self, parent):
        card = tk.Frame(parent, bg=CARD_BG, bd=1, relief="solid", highlightbackground=CARD_BORDER, padx=18, pady=12)
        card.pack(fill="x")

        # Top Bar inside Parameter Card: Title + Model Architecture Toggle
        top_bar = tk.Frame(card, bg=CARD_BG)
        top_bar.pack(fill="x", pady=(0, 10))

        tk.Label(
            top_bar,
            text="📋 Select Parameters",
            bg=CARD_BG,
            fg=PRIMARY,
            font=("Segoe UI", 12, "bold")
        ).pack(side="left")

        # Model Selection Toggle
        model_selector = tk.Frame(top_bar, bg=CARD_BG)
        model_selector.pack(side="right")

        tk.Label(
            model_selector,
            text="Active Model:",
            bg=CARD_BG,
            fg=TEXT_MUTED,
            font=("Segoe UI", 9, "bold")
        ).pack(side="left", padx=(0, 8))

        rf_rb = ttk.Radiobutton(
            model_selector,
            text="🌲 Random Forest",
            value="rf",
            variable=self.model_type_var,
            style="Model.TRadiobutton",
            command=self._on_model_toggle
        )
        rf_rb.pack(side="left", padx=4)

        ridge_rb = ttk.Radiobutton(
            model_selector,
            text="📈 Ridge Regression",
            value="linear",
            variable=self.model_type_var,
            style="Model.TRadiobutton",
            command=self._on_model_toggle
        )
        ridge_rb.pack(side="left", padx=4)

        # Dropdowns Grid (All options presented strictly in CAPITAL LETTERS)
        grid_frame = tk.Frame(card, bg=CARD_BG)
        grid_frame.pack(fill="x")

        col_w = 24
        lbl_font = ("Segoe UI", 9, "bold")

        # State
        col0 = tk.Frame(grid_frame, bg=CARD_BG)
        col0.pack(side="left", expand=True, fill="x", padx=6)
        tk.Label(col0, text="📍 STATE", bg=CARD_BG, fg=TEXT_DARK, font=lbl_font).pack(anchor="w", pady=(0, 2))
        self.state_combo = ttk.Combobox(col0, textvariable=self.state_var, values=self.states, state="readonly", width=col_w)
        self.state_combo.pack(fill="x")
        self.state_combo.bind("<<ComboboxSelected>>", self._on_state_select)

        # District
        col1 = tk.Frame(grid_frame, bg=CARD_BG)
        col1.pack(side="left", expand=True, fill="x", padx=6)
        tk.Label(col1, text="🏙 DISTRICT", bg=CARD_BG, fg=TEXT_DARK, font=lbl_font).pack(anchor="w", pady=(0, 2))
        self.district_combo = ttk.Combobox(col1, textvariable=self.district_var, state="readonly", width=col_w)
        self.district_combo.pack(fill="x")
        self.district_combo.bind("<<ComboboxSelected>>", self._on_district_select)

        # Crop
        col2 = tk.Frame(grid_frame, bg=CARD_BG)
        col2.pack(side="left", expand=True, fill="x", padx=6)
        tk.Label(col2, text="🌾 CROP", bg=CARD_BG, fg=TEXT_DARK, font=lbl_font).pack(anchor="w", pady=(0, 2))
        self.crop_combo = ttk.Combobox(col2, textvariable=self.crop_var, state="readonly", width=col_w)
        self.crop_combo.pack(fill="x")
        self.crop_combo.bind("<<ComboboxSelected>>", self._on_crop_select)

        # Season
        col3 = tk.Frame(grid_frame, bg=CARD_BG)
        col3.pack(side="left", expand=True, fill="x", padx=6)
        tk.Label(col3, text="☀ SEASON", bg=CARD_BG, fg=TEXT_DARK, font=lbl_font).pack(anchor="w", pady=(0, 2))
        self.season_combo = ttk.Combobox(col3, textvariable=self.season_var, state="readonly", width=col_w)
        self.season_combo.pack(fill="x")
        self.season_combo.bind("<<ComboboxSelected>>", self._on_season_select)

        # Context Guidance Row
        self.record_cue_label = tk.Label(
            card,
            text="Select State, District, Crop, and Season to predict yield",
            bg=CARD_BG,
            fg=TEXT_MUTED,
            font=("Segoe UI", 9)
        )
        self.record_cue_label.pack(anchor="w", pady=(8, 0))

    def _build_progress_card(self, parent):
        """Dedicated execution progress and status tracker."""
        self.progress_card = tk.Frame(parent, bg=CARD_BG, bd=1, relief="solid", highlightbackground=CARD_BORDER, padx=18, pady=10)
        self.progress_card.pack(fill="x", pady=(8, 2))

        top_row = tk.Frame(self.progress_card, bg=CARD_BG)
        top_row.pack(fill="x")

        tk.Label(
            top_row,
            text="⚙ Prediction Progress",
            bg=CARD_BG,
            fg=PRIMARY,
            font=("Segoe UI", 10, "bold")
        ).pack(side="left")

        self.progress_pill = tk.Label(
            top_row,
            text="Ready",
            bg="#E8F5E9",
            fg=SECONDARY,
            font=("Segoe UI", 8, "bold"),
            padx=8,
            pady=2
        )
        self.progress_pill.pack(side="right")

        self.progress_status_label = tk.Label(
            self.progress_card,
            text="Ready. Select parameters above and click 'Predict Yield'.",
            bg=CARD_BG,
            fg=TEXT_MUTED,
            font=("Segoe UI", 9)
        )
        self.progress_status_label.pack(anchor="w", pady=(4, 6))

        self.progressbar = ttk.Progressbar(
            self.progress_card,
            style="Green.Horizontal.TProgressbar",
            orient="horizontal",
            mode="determinate"
        )
        self.progressbar.pack(fill="x", pady=(0, 2))

    def _build_prediction_card(self, parent):
        card = tk.Frame(parent, bg=CARD_BG, bd=1, relief="solid", highlightbackground=CARD_BORDER, padx=22, pady=16)
        card.pack(side="left", fill="both", expand=True, padx=(0, 8))

        top_header = tk.Frame(card, bg=CARD_BG)
        top_header.pack(fill="x")

        tk.Label(
            top_header,
            text="🌾 Predicted Yield",
            bg=CARD_BG,
            fg=PRIMARY,
            font=("Segoe UI", 13, "bold")
        ).pack(side="left")

        self.model_badge_label = tk.Label(
            top_header,
            text="Model: Random Forest",
            bg="#E8F5E9",
            fg=SECONDARY,
            font=("Segoe UI", 9, "bold"),
            padx=10,
            pady=3
        )
        self.model_badge_label.pack(side="right")

        # Yield Value Center Display
        val_box = tk.Frame(card, bg=CARD_BG)
        val_box.pack(anchor="center", pady=(16, 6))

        self.yield_display = tk.Label(
            val_box,
            text="--",
            bg=CARD_BG,
            fg=SECONDARY,
            font=("Segoe UI", 48, "bold")
        )
        self.yield_display.pack()

        self.unit_label = tk.Label(
            val_box,
            text="Tonnes / Hectare",
            bg=CARD_BG,
            fg=TEXT_MUTED,
            font=("Segoe UI", 12, "bold")
        )
        self.unit_label.pack()

        self.conversion_label = tk.Label(
            val_box,
            text="Awaiting input",
            bg=CARD_BG,
            fg=TEXT_MUTED,
            font=("Segoe UI", 9)
        )
        self.conversion_label.pack(pady=(2, 0))

        # Yield Potential Pill
        self.rating_pill = tk.Label(
            card,
            text="Ready",
            bg="#ECEFF1",
            fg="#546E7A",
            font=("Segoe UI", 9, "bold"),
            padx=16,
            pady=4
        )
        self.rating_pill.pack(anchor="center", pady=(8, 12))

        # Bottom Context Notes
        self.context_desc_label = tk.Label(
            card,
            text="Select region and crop parameters above to predict yield.",
            bg=CARD_BG,
            fg=TEXT_MUTED,
            font=("Segoe UI", 9),
            justify="center"
        )
        self.context_desc_label.pack(fill="x", pady=(4, 0))

    def _build_performance_card(self, parent):
        card = tk.Frame(parent, bg=CARD_BG, bd=1, relief="solid", highlightbackground=CARD_BORDER, padx=22, pady=16)
        card.pack(side="right", fill="both", expand=True, padx=(8, 0))

        tk.Label(
            card,
            text="📊 Model Performance",
            bg=CARD_BG,
            fg=PRIMARY,
            font=("Segoe UI", 13, "bold")
        ).pack(anchor="w")

        # Metrics 3-Card Row
        metric_container = tk.Frame(card, bg=CARD_BG)
        metric_container.pack(fill="x", pady=(14, 10))

        def create_metric_tile(parent, title, color):
            tile = tk.Frame(parent, bg="#F9FAF9", bd=1, relief="solid", highlightbackground=CARD_BORDER, padx=12, pady=12)
            tile.pack(side="left", expand=True, fill="both", padx=4)

            tk.Label(tile, text=title, bg="#F9FAF9", fg=color, font=("Segoe UI", 10, "bold")).pack(anchor="w")
            val_lbl = tk.Label(tile, text="--", bg="#F9FAF9", fg=TEXT_DARK, font=("Segoe UI", 20, "bold"))
            val_lbl.pack(anchor="w", pady=(3, 0))

            desc_lbl = tk.Label(tile, text="", bg="#F9FAF9", fg=TEXT_MUTED, font=("Segoe UI", 8))
            desc_lbl.pack(anchor="w")
            return val_lbl, desc_lbl

        self.r2_value, self.r2_desc = create_metric_tile(metric_container, "R² Score", SECONDARY)
        self.rmse_value, self.rmse_desc = create_metric_tile(metric_container, "RMSE", ACCENT_BLUE)
        self.mae_value, self.mae_desc = create_metric_tile(metric_container, "MAE", ACCENT_AMBER)

        self.r2_desc.config(text="Goodness of fit")
        self.rmse_desc.config(text="Root mean squared error")
        self.mae_desc.config(text="Mean absolute error")

        # Comparison Preview Box (RF vs Ridge)
        ttk.Separator(card, orient="horizontal").pack(fill="x", pady=8)

        self.comp_frame = tk.Frame(card, bg="#F1F6F1", bd=1, relief="solid", highlightbackground=CARD_BORDER, padx=14, pady=10)
        self.comp_frame.pack(fill="x", expand=True)

        tk.Label(
            self.comp_frame,
            text="⚡ Model Comparison",
            bg="#F1F6F1",
            fg=PRIMARY,
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w")

        self.comp_text_label = tk.Label(
            self.comp_frame,
            text="Click 'Compare Models' to compare Random Forest and Ridge Regression.",
            bg="#F1F6F1",
            fg=TEXT_MUTED,
            font=("Segoe UI", 9),
            justify="left"
        )
        self.comp_text_label.pack(anchor="w", pady=(4, 0))

    def _build_action_bar(self, parent):
        bar = tk.Frame(parent, bg=BG_COLOR)
        bar.pack(fill="x", pady=(8, 2))

        btn_common = {
            "font": ("Segoe UI", 10, "bold"),
            "bd": 0,
            "cursor": "hand2",
            "padx": 16,
            "pady": 8,
            "relief": "flat"
        }

        # 1. Predict
        self.predict_btn = tk.Button(
            bar,
            text="🌲 Predict Yield (Enter)",
            bg=SECONDARY,
            fg="white",
            activebackground=PRIMARY,
            activeforeground="white",
            command=self.run_prediction,
            **btn_common
        )
        self.predict_btn.pack(side="left", padx=4)

        # 2. Compare Both
        self.compare_btn = tk.Button(
            bar,
            text="⚡ Compare Models (RF vs Ridge)",
            bg=ACCENT_BLUE,
            fg="white",
            activebackground="#0D47A1",
            activeforeground="white",
            command=self.run_comparison,
            **btn_common
        )
        self.compare_btn.pack(side="left", padx=4)

        # 3. Detailed View
        self.detail_btn = tk.Button(
            bar,
            text="📊 Detailed View (F1)",
            bg="#37474F",
            fg="white",
            activebackground="#263238",
            activeforeground="white",
            command=self.open_detailed_view,
            **btn_common
        )
        self.detail_btn.pack(side="left", padx=4)

        # 4. Reset
        self.reset_btn = tk.Button(
            bar,
            text="🔄 Reset (Esc)",
            bg="#78909C",
            fg="white",
            activebackground="#546E7A",
            activeforeground="white",
            command=self.reset_form,
            **btn_common
        )
        self.reset_btn.pack(side="left", padx=4)

        # 5. Exit
        self.exit_btn = tk.Button(
            bar,
            text="✕ Exit",
            bg="#D32F2F",
            fg="white",
            activebackground="#B71C1C",
            activeforeground="white",
            command=self.root.destroy,
            **btn_common
        )
        self.exit_btn.pack(side="right", padx=4)

        # Hover effects
        for btn in [self.predict_btn, self.compare_btn, self.detail_btn, self.reset_btn, self.exit_btn]:
            normal_bg = btn.cget("bg")
            btn.bind("<Enter>", lambda e, b=btn, c=normal_bg: b.config(relief="groove"))
            btn.bind("<Leave>", lambda e, b=btn: b.config(relief="flat"))

    def _bind_shortcuts(self):
        self.root.bind("<Return>", lambda e: self.run_prediction())
        self.root.bind("<Shift-Return>", lambda e: self.run_comparison())
        self.root.bind("<F1>", lambda e: self.open_detailed_view())
        self.root.bind("<Control-d>", lambda e: self.open_detailed_view())
        self.root.bind("<Escape>", lambda e: self.reset_form())

    # Cascading Dropdown Handlers (Strictly CAPITAL LETTERS & filtered to available data)
    def _on_state_select(self, event=None):
        state = self.state_var.get().upper()
        districts = sorted(self.df[self.df["STATE_UPPER"] == state]["DISTRICT_UPPER"].dropna().unique())

        self.district_combo["values"] = districts
        self.district_var.set("")
        self.crop_var.set("")
        self.season_var.set("")
        self.crop_combo["values"] = []
        self.season_combo["values"] = []
        self.record_cue_label.config(text=f"📍 STATE: {state}  →  Select DISTRICT", fg=PRIMARY)

    def _on_district_select(self, event=None):
        state = self.state_var.get().upper()
        district = self.district_var.get().upper()

        crops = sorted(self.df[
            (self.df["STATE_UPPER"] == state) &
            (self.df["DISTRICT_UPPER"] == district)
        ]["CROP_UPPER"].dropna().unique())

        self.crop_combo["values"] = crops
        self.crop_var.set("")
        self.season_var.set("")
        self.season_combo["values"] = []
        self.record_cue_label.config(text=f"🏙 {district}, {state}  →  {len(crops)} crops available. Select CROP", fg=PRIMARY)

    def _on_crop_select(self, event=None):
        state = self.state_var.get().upper()
        district = self.district_var.get().upper()
        crop = self.crop_var.get().upper()

        seasons = sorted(self.df[
            (self.df["STATE_UPPER"] == state) &
            (self.df["DISTRICT_UPPER"] == district) &
            (self.df["CROP_UPPER"] == crop)
        ]["SEASON_UPPER"].dropna().unique())

        self.season_combo["values"] = seasons
        self.season_var.set("")
        self.record_cue_label.config(text=f"🌾 {crop}  →  {len(seasons)} seasons available. Select SEASON", fg=PRIMARY)

    def _on_season_select(self, event=None):
        state = self.state_var.get().upper()
        district = self.district_var.get().upper()
        crop = self.crop_var.get().upper()
        season = self.season_var.get().upper()

        matched = self.df[
            (self.df["STATE_UPPER"] == state) &
            (self.df["DISTRICT_UPPER"] == district) &
            (self.df["CROP_UPPER"] == crop) &
            (self.df["SEASON_UPPER"] == season)
        ]
        count = len(matched)
        if count > 0:
            avg_yield = matched["yield"].mean()
            min_yr = matched["year"].min()
            max_yr = matched["year"].max()
            rain_val = matched["average_rainfall"].mean() if "average_rainfall" in matched.columns else None
            rain_str = f"Avg Rain: {rain_val:.0f} mm  •  " if rain_val is not None else ""
            self.record_cue_label.config(
                text=(
                    f"✓ Configuration Ready: {count} historical records ({min_yr}–{max_yr})  •  "
                    f"District Avg: {avg_yield:.2f} t/ha  •  "
                    f"{rain_str}Press 'Predict Yield' or Enter."
                ),
                fg=SECONDARY
            )
        else:
            self.record_cue_label.config(
                text="✓ Configuration Ready. Press 'Predict Yield' or Enter.",
                fg=SECONDARY
            )

    def _on_model_toggle(self):
        m = self.model_type_var.get()
        name = "Random Forest Regressor" if m == "rf" else "Ridge Regression (L2)"
        self.predict_btn.config(text=f"🌲 Predict ({'RF' if m == 'rf' else 'Ridge'}) [Enter]")
        self.model_badge_label.config(
            text=f"Model: {name}",
            bg="#E8F5E9" if m == "rf" else "#E3F2FD",
            fg=SECONDARY if m == "rf" else ACCENT_BLUE
        )
        # If inputs already filled, re-predict with selected model
        if self._inputs_valid(silent=True):
            self.run_prediction()

    def _inputs_valid(self, silent=False):
        missing = []
        if not self.state_var.get():
            missing.append("STATE")
        if not self.district_var.get():
            missing.append("DISTRICT")
        if not self.crop_var.get():
            missing.append("CROP")
        if not self.season_var.get():
            missing.append("SEASON")

        if missing:
            if not silent:
                messagebox.showwarning(
                    "Missing Parameters",
                    f"Please complete all parameter selections:\n• " + "\n• ".join(missing),
                    parent=self.root
                )
            return False
        return True

    def run_prediction(self):
        if not self._inputs_valid():
            return

        state = self.state_var.get().upper()
        district = self.district_var.get().upper()
        crop = self.crop_var.get().upper()
        season = self.season_var.get().upper()
        model_type = self.model_type_var.get()
        model_name = "Random Forest" if model_type == "rf" else "Ridge Regression"

        # Disable buttons temporarily during stage animation
        for btn in [self.predict_btn, self.compare_btn, self.detail_btn, self.reset_btn]:
            btn.config(state="disabled")

        self.root.config(cursor="wait")
        self.progress_pill.config(text="Running...", bg="#FFF8E1", fg=ACCENT_AMBER)
        self.progressbar["value"] = 0

        stages = [
            (f"Loading data for {crop.title()} in {district.title()}...", 25),
            (f"Loading {model_name} model...", 50),
            (f"Calculating historical averages...", 75),
            (f"Generating yield prediction...", 90),
        ]

        def _execute_stage(stage_idx):
            if stage_idx < len(stages):
                msg, val = stages[stage_idx]
                self.progress_status_label.config(text=msg)
                self.progressbar["value"] = val
                self.root.after(75, lambda: _execute_stage(stage_idx + 1))
            else:
                self._finish_prediction(state, district, crop, season, model_type, model_name)

        _execute_stage(0)

    def _finish_prediction(self, state, district, crop, season, model_type, model_name):
        try:
            res = predict_crop(state, district, crop, season, model_type=model_type)
            self.last_prediction_data = res

            val = res["predicted_yield"]
            quintals_ha = val * 10.0
            quintals_acre = val * 4.047
            kg = val * 1000.0

            # Determine yield status tier
            if val >= 3.0:
                color = SECONDARY
                rating_text = "High Yield"
                rating_bg = "#E8F5E9"
                rating_fg = "#2E7D32"
            elif val >= 1.8:
                color = "#FB8C00"
                rating_text = "Moderate Yield"
                rating_bg = "#FFF8E1"
                rating_fg = "#F57F17"
            else:
                color = "#D32F2F"
                rating_text = "Low Yield"
                rating_bg = "#FFEBEE"
                rating_fg = "#C62828"

            self.yield_display.config(text=f"{val:.3f}", fg=color)
            self.conversion_label.config(
                text=f"≈ {quintals_ha:.1f} Quintals/Ha   •   {quintals_acre:.1f} Quintals/Acre   •   {kg:.0f} kg/Ha"
            )
            self.rating_pill.config(text=rating_text, bg=rating_bg, fg=rating_fg)

            self.r2_value.config(text=f"{res['r2']:.3f}")
            self.rmse_value.config(text=f"{res['rmse']:.3f}")
            self.mae_value.config(text=f"{res['mae']:.3f}")

            # Calculate comparison against historical average
            hist_comparison = ""
            matched = self.df[
                (self.df["STATE_UPPER"] == state) &
                (self.df["DISTRICT_UPPER"] == district) &
                (self.df["CROP_UPPER"] == crop) &
                (self.df["SEASON_UPPER"] == season)
            ]
            if len(matched) > 0:
                hist_mean = float(matched["yield"].mean())
                diff_val = val - hist_mean
                diff_pct = (diff_val / hist_mean) * 100
                diff_sign = "+" if diff_val >= 0 else ""
                hist_comparison = f" ({diff_sign}{diff_pct:.1f}% vs district 5-yr avg of {hist_mean:.2f} t/ha)"

            self.context_desc_label.config(
                text=(
                    f"Predicted yield for {crop.title()} in {district.title()}, {state.title()} ({season.title()} season).\n"
                    f"Validation error on latest harvest: ±{res['prediction_error']:.3f} t/ha{hist_comparison}."
                )
            )

            # Auto-fetch alternate model prediction for the comparison box
            alt_type = "linear" if model_type == "rf" else "rf"
            alt_res = predict_crop(state, district, crop, season, model_type=alt_type)
            alt_name = "Ridge" if alt_type == "linear" else "Random Forest"
            diff = abs(val - alt_res["predicted_yield"])

            self.comp_text_label.config(
                text=(
                    f"• {model_name}: {val:.3f} t/ha (R² = {res['r2']:.3f})\n"
                    f"• {alt_name}: {alt_res['predicted_yield']:.3f} t/ha (R² = {alt_res['r2']:.3f})\n"
                    f"• Difference: {diff:.3f} t/ha"
                )
            )

            self.progressbar["value"] = 100
            self.progress_pill.config(text="Done", bg="#E8F5E9", fg=SECONDARY)
            self.progress_status_label.config(
                text=f"✓ Prediction completed with {model_name} (R² = {res['r2']:.3f})"
            )
            self.status_label.config(
                text=f"Ready  •  Prediction completed ({model_name}, R² = {res['r2']:.3f})"
            )

        except Exception as e:
            self.progressbar["value"] = 0
            self.progress_pill.config(text="ERROR", bg="#FFEBEE", fg="#C62828")
            self.progress_status_label.config(text=f"Execution error: {str(e)}")
            messagebox.showerror("Prediction Error", str(e), parent=self.root)
            self.status_label.config(text="Prediction failed.")
        finally:
            self.root.config(cursor="")
            for btn in [self.predict_btn, self.compare_btn, self.detail_btn, self.reset_btn]:
                btn.config(state="normal")

    def run_comparison(self):
        """Run both Random Forest and Ridge Regression and present an explicit modal comparison."""
        if not self._inputs_valid():
            return

        state = self.state_var.get().upper()
        district = self.district_var.get().upper()
        crop = self.crop_var.get().upper()
        season = self.season_var.get().upper()

        self.root.config(cursor="wait")
        self.status_label.config(text="Comparing Random Forest vs Ridge Regression...")
        self.root.update_idletasks()

        try:
            comp = compare_models(state, district, crop, season)

            # Build clean top-level modal dialog for side-by-side inspection
            dialog = tk.Toplevel(self.root)
            dialog.title(f"Model Comparison: {crop.title()} ({district.title()})")
            dialog.configure(bg=BG_COLOR)
            dialog.bind("<Escape>", lambda e: dialog.destroy())

            # Responsive centered geometry avoiding screen edge clipping
            dialog.update_idletasks()
            sw = dialog.winfo_screenwidth()
            sh = dialog.winfo_screenheight()
            dlg_w = min(800, max(680, sw - 40))
            dlg_h = min(540, max(440, sh - 100))
            pos_x = max(10, (sw - dlg_w) // 2)
            pos_y = max(10, (sh - dlg_h) // 2)
            dialog.geometry(f"{dlg_w}x{dlg_h}+{pos_x}+{pos_y}")
            dialog.minsize(640, 420)
            dialog.resizable(True, True)
            dialog.transient(self.root)
            dialog.focus_force()

            # Top Header Bar
            d_head = tk.Frame(dialog, bg=PRIMARY, height=54)
            d_head.pack(fill="x")
            d_head.pack_propagate(False)

            tk.Label(
                d_head,
                text=f"⚡ Model Comparison: {crop.title()} ({season.title()} Season)",
                bg=PRIMARY,
                fg="white",
                font=("Segoe UI", 12, "bold")
            ).pack(side="left", padx=18, pady=14)

            # Docked Bottom Action Bar (One single, prominent, properly visible close button)
            btn_bar = tk.Frame(dialog, bg="#E8EFE8", bd=1, relief="solid", highlightbackground=CARD_BORDER, padx=20, pady=10)
            btn_bar.pack(side="bottom", fill="x")

            tk.Label(
                btn_bar,
                text="Press Esc or click Close to return to the main window",
                bg="#E8EFE8",
                fg=TEXT_MUTED,
                font=("Segoe UI", 9)
            ).pack(side="left")

            tk.Button(
                btn_bar,
                text="✕ Close",
                bg="#C62828",
                fg="white",
                activebackground="#B71C1C",
                activeforeground="white",
                font=("Segoe UI", 10, "bold"),
                bd=0,
                padx=26,
                pady=7,
                cursor="hand2",
                command=dialog.destroy
            ).pack(side="right")

            # Main Cards Container
            container = tk.Frame(dialog, bg=BG_COLOR, padx=18, pady=14)
            container.pack(fill="both", expand=True)

            cards_box = tk.Frame(container, bg=BG_COLOR)
            cards_box.pack(fill="x")

            # RF Card
            rf_box = tk.Frame(cards_box, bg=CARD_BG, bd=1, relief="solid", highlightbackground=CARD_BORDER, padx=16, pady=14)
            rf_box.pack(side="left", expand=True, fill="both", padx=(0, 8))

            rf_header = tk.Frame(rf_box, bg=CARD_BG)
            rf_header.pack(fill="x", anchor="w")
            tk.Label(rf_header, text="🌲 RANDOM FOREST", bg="#E8F5E9", fg=SECONDARY, font=("Segoe UI", 9, "bold"), padx=8, pady=3).pack(side="left")

            tk.Label(rf_box, text=f"{comp['rf']['predicted_yield']:.3f} t/ha", bg=CARD_BG, fg=TEXT_DARK, font=("Segoe UI", 26, "bold")).pack(anchor="w", pady=(10, 4))
            tk.Label(rf_box, text=f"R² Score:  {comp['rf']['r2']:.4f}", bg=CARD_BG, fg=TEXT_MUTED, font=("Segoe UI", 10)).pack(anchor="w")
            tk.Label(rf_box, text=f"RMSE:      {comp['rf']['rmse']:.4f}", bg=CARD_BG, fg=TEXT_MUTED, font=("Segoe UI", 10)).pack(anchor="w")
            tk.Label(rf_box, text=f"MAE:       {comp['rf']['mae']:.4f}", bg=CARD_BG, fg=TEXT_MUTED, font=("Segoe UI", 10)).pack(anchor="w")

            # Ridge Card
            lin_box = tk.Frame(cards_box, bg=CARD_BG, bd=1, relief="solid", highlightbackground=CARD_BORDER, padx=16, pady=14)
            lin_box.pack(side="right", expand=True, fill="both", padx=(8, 0))

            lin_header = tk.Frame(lin_box, bg=CARD_BG)
            lin_header.pack(fill="x", anchor="w")
            tk.Label(lin_header, text="📈 RIDGE REGRESSION", bg="#E3F2FD", fg=ACCENT_BLUE, font=("Segoe UI", 9, "bold"), padx=8, pady=3).pack(side="left")

            tk.Label(lin_box, text=f"{comp['ridge']['predicted_yield']:.3f} t/ha", bg=CARD_BG, fg=TEXT_DARK, font=("Segoe UI", 26, "bold")).pack(anchor="w", pady=(10, 4))
            tk.Label(lin_box, text=f"R² Score:  {comp['ridge']['r2']:.4f}", bg=CARD_BG, fg=TEXT_MUTED, font=("Segoe UI", 10)).pack(anchor="w")
            tk.Label(lin_box, text=f"RMSE:      {comp['ridge']['rmse']:.4f}", bg=CARD_BG, fg=TEXT_MUTED, font=("Segoe UI", 10)).pack(anchor="w")
            tk.Label(lin_box, text=f"MAE:       {comp['ridge']['mae']:.4f}", bg=CARD_BG, fg=TEXT_MUTED, font=("Segoe UI", 10)).pack(anchor="w")

            # Comparison Summary Frame
            syn_frame = tk.Frame(container, bg="#E8F5E9", bd=1, relief="solid", highlightbackground="#C8E6C9", padx=16, pady=12)
            syn_frame.pack(fill="x", pady=(14, 0))

            tk.Label(syn_frame, text="Comparison Summary", bg="#E8F5E9", fg=PRIMARY, font=("Segoe UI", 10, "bold")).pack(anchor="w")

            higher_model = "Random Forest" if comp["rf"]["r2"] >= comp["ridge"]["r2"] else "Ridge Regression"
            higher_r2 = max(comp["rf"]["r2"], comp["ridge"]["r2"])

            tk.Label(
                syn_frame,
                text=(
                    f"• Average of both models: {comp['consensus_yield']:.3f} Tonnes / Hectare\n"
                    f"• Difference: {comp['difference']:.3f} t/ha ({comp['relative_diff_pct']:.1f}%)\n"
                    f"• Better fit on test data: {higher_model} (R² = {higher_r2:.4f})"
                ),
                bg="#E8F5E9",
                fg=TEXT_DARK,
                font=("Segoe UI", 9),
                justify="left"
            ).pack(anchor="w", pady=(6, 0))

            self.status_label.config(text="✓ Dual-model comparison completed.")

        except Exception as e:
            messagebox.showerror("Comparison Error", str(e), parent=self.root)
        finally:
            self.root.config(cursor="")

    def open_detailed_view(self):
        if not self._inputs_valid():
            return

        model_type = self.model_type_var.get()
        try:
            detailed_view.show_window(
                self.state_var.get().upper(),
                self.district_var.get().upper(),
                self.crop_var.get().upper(),
                self.season_var.get().upper(),
                model_type=model_type
            )
        except Exception as e:
            messagebox.showerror("Detailed View Error", str(e), parent=self.root)

    def reset_form(self):
        self.state_var.set("")
        self.district_var.set("")
        self.crop_var.set("")
        self.season_var.set("")

        self.district_combo["values"] = []
        self.crop_combo["values"] = []
        self.season_combo["values"] = []

        self.yield_display.config(text="--", fg=SECONDARY)
        self.conversion_label.config(text="Awaiting input")
        self.rating_pill.config(text="Ready", bg="#ECEFF1", fg="#546E7A")

        self.r2_value.config(text="--")
        self.rmse_value.config(text="--")
        self.mae_value.config(text="--")

        self.progressbar["value"] = 0
        self.progress_pill.config(text="Ready", bg="#E8F5E9", fg=SECONDARY)
        self.progress_status_label.config(text="Ready. Select parameters above and click 'Predict Yield'.")

        self.context_desc_label.config(
            text="Select region and crop parameters above to predict yield."
        )
        self.comp_text_label.config(
            text="Click 'Compare Models' to compare Random Forest and Ridge Regression."
        )
        self.record_cue_label.config(
            text="Select State, District, Crop, and Season to predict yield",
            fg=TEXT_MUTED
        )
        self.status_label.config(text="Form reset. Ready for new input.")


def main():
    root = tk.Tk()
    app = CropYieldApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
