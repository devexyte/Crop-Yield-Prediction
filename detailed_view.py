import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Circle
from crop_yield_prediction import detailed_prediction
import report_exporter

PRIMARY_COLOR = "#1B4D2E"
SECONDARY_COLOR = "#2E7D32"
BG_COLOR = "#F4F7F4"
CARD_BG = "#FFFFFF"
CARD_BORDER = "#D8E2D8"
TEXT_COLOR = "#1C251D"
MUTED_TEXT = "#5C6B5E"
ACCENT_BLUE = "#1565C0"
ACCENT_AMBER = "#E65100"


def setup_treeview_style():
    """Configure modern styling for ttk.Treeview widgets."""
    style = ttk.Style()
    style.theme_use("clam")

    style.configure(
        "Modern.Treeview.Heading",
        background="#205833",
        foreground="white",
        font=("Segoe UI", 10, "bold"),
        padding=(8, 6),
        relief="flat"
    )
    style.map(
        "Modern.Treeview.Heading",
        background=[("active", "#1B4D2E")]
    )

    style.configure(
        "Modern.Treeview",
        background="white",
        foreground=TEXT_COLOR,
        rowheight=30,
        font=("Segoe UI", 9),
        fieldbackground="white",
        borderwidth=0
    )
    style.map(
        "Modern.Treeview",
        background=[("selected", "#C8E6C9")],
        foreground=[("selected", "#1B4D2E")]
    )


def create_styled_tree(parent, dataframe):
    """Render a DataFrame as a clean, scrollable Treeview with zebra rows."""
    setup_treeview_style()

    container = tk.Frame(parent, bg=CARD_BG, bd=1, relief="solid", highlightbackground=CARD_BORDER)
    container.pack(fill="both", expand=True, padx=2, pady=2)

    tree = ttk.Treeview(container, style="Modern.Treeview", show="headings")
    tree["columns"] = list(dataframe.columns)

    for col in dataframe.columns:
        tree.heading(col, text=str(col))
        sample_len = max(len(str(col)), max([len(str(val)) for val in dataframe[col].head(15)], default=8))
        col_width = max(110, min(sample_len * 12 + 25, 260))
        tree.column(col, width=col_width, anchor="center")

    for i, (_, row) in enumerate(dataframe.iterrows()):
        formatted_vals = []
        for val in row:
            if isinstance(val, float):
                formatted_vals.append(f"{val:.3f}" if abs(val) < 100 else f"{val:.1f}")
            else:
                formatted_vals.append(str(val))

        tag = "evenrow" if i % 2 == 0 else "oddrow"
        tree.insert("", "end", values=formatted_vals, tags=(tag,))

    tree.tag_configure("evenrow", background="#F8FAF8")
    tree.tag_configure("oddrow", background="#FFFFFF")

    scroll_y = ttk.Scrollbar(container, orient="vertical", command=tree.yview)
    scroll_x = ttk.Scrollbar(container, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)

    tree.grid(row=0, column=0, sticky="nsew")
    scroll_y.grid(row=0, column=1, sticky="ns")
    scroll_x.grid(row=1, column=0, sticky="ew")

    container.grid_rowconfigure(0, weight=1)
    container.grid_columnconfigure(0, weight=1)

    def _on_shift_wheel(e):
        tree.xview_scroll(int(-1 * (e.delta / 120)), "units")

    tree.bind("<Shift-MouseWheel>", _on_shift_wheel)

    # Register as scroll target for active tab
    parent._scroll_target = tree
    return tree, container


def create_scrollable_container(parent):
    """
    Build a scrollable frame with a Canvas and Scrollbar configured for web-like smooth scrolling.
    """
    canvas = tk.Canvas(parent, bg=BG_COLOR, highlightthickness=0, yscrollincrement=5)
    scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
    scroll_frame = tk.Frame(canvas, bg=BG_COLOR)

    def _update_scrollregion(e=None):
        canvas.update_idletasks()
        bbox = canvas.bbox("all")
        if bbox:
            canvas.configure(scrollregion=(bbox[0], bbox[1], bbox[2], bbox[3] + 40))

    scroll_frame.bind("<Configure>", _update_scrollregion)
    window_id = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")

    def _on_canvas_configure(e):
        canvas.itemconfig(window_id, width=e.width)

    canvas.bind("<Configure>", _on_canvas_configure)
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Set scroll target on parent tab
    parent._scroll_target = canvas
    scroll_frame._parent_canvas = canvas
    return scroll_frame


def create_card_frame(parent, title=None, padx=25, pady=10):
    """Reusable styled card with optional header bar and border."""
    wrapper = tk.Frame(parent, bg=BG_COLOR)
    wrapper.pack(fill="x", padx=padx, pady=pady)

    card = tk.Frame(wrapper, bg=CARD_BG, bd=1, relief="solid", highlightbackground=CARD_BORDER)
    card.pack(fill="both", expand=True)

    if title:
        header_bar = tk.Frame(card, bg="#F1F6F1", height=36)
        header_bar.pack(fill="x")
        header_bar.pack_propagate(False)

        tk.Label(
            header_bar,
            text=title,
            bg="#F1F6F1",
            fg=PRIMARY_COLOR,
            font=("Segoe UI", 11, "bold")
        ).pack(side="left", padx=16, pady=6)

    content_frame = tk.Frame(card, bg=CARD_BG, padx=16, pady=14)
    content_frame.pack(fill="both", expand=True)
    return content_frame


def setup_window_scrolling(window, notebook):
    """
    Attach universal mouse wheel and keyboard scrolling to the detailed view window.
    Applies fluid, website-style scroll distance per wheel notch.
    """
    def _on_mousewheel(event):
        try:
            active_tab_id = notebook.select()
            if not active_tab_id:
                return
            active_tab = notebook.nametowidget(active_tab_id)
            target = getattr(active_tab, "_scroll_target", None)
            if target and hasattr(target, "yview_scroll"):
                # Natural web-like scroll travel (4-5 units per notch)
                step = int(-1 * (event.delta / 120) * 5)
                target.yview_scroll(step, "units")
                return "break"
        except Exception:
            pass

    def _on_key_scroll(event):
        try:
            active_tab_id = notebook.select()
            if not active_tab_id:
                return
            active_tab = notebook.nametowidget(active_tab_id)
            target = getattr(active_tab, "_scroll_target", None)
            if not target or not hasattr(target, "yview_scroll"):
                return

            if event.keysym in ["Page_Down", "space"]:
                target.yview_scroll(8, "pages")
                return "break"
            elif event.keysym == "Page_Up":
                target.yview_scroll(-8, "pages")
                return "break"
            elif event.keysym == "Down":
                target.yview_scroll(3, "units")
                return "break"
            elif event.keysym == "Up":
                target.yview_scroll(-3, "units")
                return "break"
        except Exception:
            pass

    window.bind_all("<MouseWheel>", _on_mousewheel)
    window.bind("<Page_Down>", _on_key_scroll)
    window.bind("<Page_Up>", _on_key_scroll)
    window.bind("<Down>", _on_key_scroll)
    window.bind("<Up>", _on_key_scroll)
    window.bind("<space>", _on_key_scroll)


def show_window(state, district, crop, season, model_type="rf"):
    """Open the comprehensive analytics and reporting window."""
    result = detailed_prediction(state, district, crop, season, model_type=model_type)

    prediction = result["prediction"]
    performance = result["performance"]
    summary = result["summary"]
    history = result["history"]
    weather = result["weather"]
    features = result["features"]
    crop_yield_series = result.get("crop_yield_series", np.array([]))

    window = tk.Toplevel()
    model_name = prediction.get("model_type", "Random Forest")
    crop_upper = str(crop).strip().upper()
    district_upper = str(district).strip().upper()
    state_upper = str(state).strip().upper()

    window.title(f"Crop Yield Report - {crop.title()} ({district.title()}) | {model_name}")
    
    # Adaptive geometry centered on screen
    window.update_idletasks()
    sw = window.winfo_screenwidth()
    sh = window.winfo_screenheight()
    win_w = min(1260, sw - 40)
    win_h = min(820, sh - 80)
    win_x = max(10, (sw - win_w) // 2)
    win_y = max(10, (sh - win_h) // 2)
    window.geometry(f"{win_w}x{win_h}+{win_x}+{win_y}")
    window.minsize(980, 620)
    window.configure(bg=BG_COLOR)

    # Clean memory release on exit
    active_figures = []

    def close_report():
        try:
            window.unbind_all("<MouseWheel>")
        except Exception:
            pass
        for fig in active_figures:
            fig.clf()
        active_figures.clear()
        window.destroy()

    window.protocol("WM_DELETE_WINDOW", close_report)
    window.bind("<Escape>", lambda e: close_report())

    # Header Bar (sizes naturally with padding so text is never clipped)
    header = tk.Frame(window, bg=PRIMARY_COLOR)
    header.pack(fill="x")

    title_box = tk.Frame(header, bg=PRIMARY_COLOR)
    title_box.pack(side="left", padx=24, pady=12)

    tk.Label(
        title_box,
        text=f"Crop Yield Report: {crop.title()}",
        bg=PRIMARY_COLOR,
        fg="#FFFFFF",
        font=("Segoe UI", 16, "bold")
    ).pack(anchor="w")

    tk.Label(
        title_box,
        text=f"{district.title()}, {state.title()}   •   {season.title()} Season   •   {model_name}",
        bg=PRIMARY_COLOR,
        fg="#D7ECD9",
        font=("Segoe UI", 10)
    ).pack(anchor="w", pady=(3, 0))

    # Header Action Buttons
    action_box = tk.Frame(header, bg=PRIMARY_COLOR)
    action_box.pack(side="right", padx=20, pady=12)

    def export_report(default_fmt="pdf"):
        if default_fmt == "docx":
            filetypes = [("Word Document (*.docx)", "*.docx"), ("PDF Document (*.pdf)", "*.pdf")]
            defext = ".docx"
            init_file = f"crop_yield_report_{crop.lower()}_{district.lower()}.docx"
        else:
            filetypes = [("PDF Document (*.pdf)", "*.pdf"), ("Word Document (*.docx)", "*.docx")]
            defext = ".pdf"
            init_file = f"crop_yield_report_{crop.lower()}_{district.lower()}.pdf"

        filepath = filedialog.asksaveasfilename(
            parent=window,
            defaultextension=defext,
            filetypes=filetypes,
            initialfile=init_file
        )
        if not filepath:
            return

        try:
            ext = os.path.splitext(filepath)[1].lower()
            if ext == ".docx":
                report_exporter.export_docx_report(
                    filepath, crop, district, state, season, model_name,
                    prediction, performance, summary, history, weather, features, crop_yield_series
                )
                fmt_name = "Word Document (.docx)"
            elif ext == ".txt":
                report_exporter.export_txt_report(
                    filepath, crop, district, state, season, model_name,
                    prediction, performance, summary, history, weather, features
                )
                fmt_name = "Text Summary (.txt)"
            else:
                report_exporter.export_pdf_report(
                    filepath, crop, district, state, season, model_name,
                    prediction, performance, summary, history, weather, features, crop_yield_series
                )
                fmt_name = "PDF Document (.pdf)"

            ans = messagebox.askyesno(
                "Export Successful",
                f"Advisory report saved as {fmt_name} to:\n\n{filepath}\n\nWould you like to open it now?",
                parent=window
            )
            if ans:
                try:
                    os.startfile(filepath)
                except Exception:
                    pass
        except Exception as err:
            messagebox.showerror("Export Failed", f"Could not export report:\n{str(err)}", parent=window)

    # 1. Export PDF Button
    export_pdf_btn = tk.Button(
        action_box,
        text="📄 Export PDF",
        bg="#2E7D32",
        fg="white",
        activebackground="#1B5E20",
        activeforeground="white",
        font=("Segoe UI", 9, "bold"),
        bd=0,
        padx=13,
        pady=6,
        cursor="hand2",
        command=lambda: export_report("pdf")
    )
    export_pdf_btn.pack(side="left", padx=4)

    # 2. Export Word Button
    export_docx_btn = tk.Button(
        action_box,
        text="📝 Export Word (.docx)",
        bg="#1565C0",
        fg="white",
        activebackground="#0D47A1",
        activeforeground="white",
        font=("Segoe UI", 9, "bold"),
        bd=0,
        padx=13,
        pady=6,
        cursor="hand2",
        command=lambda: export_report("docx")
    )
    export_docx_btn.pack(side="left", padx=4)

    # 3. Close Button
    close_btn = tk.Button(
        action_box,
        text="✕ Close",
        bg="#B71C1C",
        fg="white",
        activebackground="#880E4F",
        activeforeground="white",
        font=("Segoe UI", 9, "bold"),
        bd=0,
        padx=14,
        pady=6,
        cursor="hand2",
        command=close_report
    )
    close_btn.pack(side="left", padx=4)

    # Bottom Status Footer (packed before notebook to guarantee its bottom dock position)
    footer = tk.Frame(window, bg="#E8EFE8", height=28)
    footer.pack(fill="x", side="bottom")
    footer.pack_propagate(False)

    tk.Label(
        footer,
        text=f"Crop Yield Report  •  {crop.title()} in {district.title()}, {state.title()}  •  {model_name}",
        bg="#E8EFE8",
        fg=MUTED_TEXT,
        font=("Segoe UI", 9)
    ).pack(side="left", padx=20, pady=3)

    tk.Label(
        footer,
        text="Press Esc to Close",
        bg="#E8EFE8",
        fg=MUTED_TEXT,
        font=("Segoe UI", 9)
    ).pack(side="right", padx=20, pady=3)

    # Main Notebook (Tabs)
    style = ttk.Style()
    style.configure(
        "Report.TNotebook",
        background=BG_COLOR,
        borderwidth=0
    )
    style.configure(
        "Report.TNotebook.Tab",
        font=("Segoe UI", 10, "bold"),
        padding=(18, 9),
        background="#DFE8DF",
        foreground=TEXT_COLOR
    )
    style.map(
        "Report.TNotebook.Tab",
        background=[("selected", "#FFFFFF"), ("active", "#E8F0E8")],
        foreground=[("selected", PRIMARY_COLOR)]
    )

    notebook = ttk.Notebook(window, style="Report.TNotebook")
    notebook.pack(fill="both", expand=True, padx=20, pady=(12, 10))

    tab_pred = tk.Frame(notebook, bg=BG_COLOR)
    tab_graphs = tk.Frame(notebook, bg=BG_COLOR)
    tab_history = tk.Frame(notebook, bg=BG_COLOR)
    tab_weather = tk.Frame(notebook, bg=BG_COLOR)
    tab_features = tk.Frame(notebook, bg=BG_COLOR)
    tab_summary = tk.Frame(notebook, bg=BG_COLOR)

    notebook.add(tab_pred, text="  🌾 Overview  ")
    notebook.add(tab_graphs, text="  📊 Charts  ")
    notebook.add(tab_history, text="  📈 History  ")
    notebook.add(tab_weather, text="  ☁ Weather & Soil  ")
    notebook.add(tab_features, text="  📋 Features  ")
    notebook.add(tab_summary, text="  📁 Dataset Summary  ")

    # ==============================================================
    # TAB 1: OVERVIEW & METRICS
    # ==============================================================
    pred_scroll = create_scrollable_container(tab_pred)

    hero_card = create_card_frame(pred_scroll, title="🌾 Predicted Yield", padx=30, pady=12)

    pred_val = prediction["predicted_yield"]
    quintals_ha = pred_val * 10.0
    quintals_acre = pred_val * 4.047
    kg_val = pred_val * 1000.0

    if pred_val >= 3.0:
        badge_text = "High Yield"
        badge_bg = "#E8F5E9"
        badge_fg = "#2E7D32"
    elif pred_val >= 1.8:
        badge_text = "Moderate Yield"
        badge_bg = "#FFF8E1"
        badge_fg = "#F57F17"
    else:
        badge_text = "Low Yield"
        badge_bg = "#FFEBEE"
        badge_fg = "#C62828"

    yield_box = tk.Frame(hero_card, bg=CARD_BG)
    yield_box.pack(fill="x", pady=6)

    tk.Label(
        yield_box,
        text=f"{pred_val:.3f}",
        bg=CARD_BG,
        fg=SECONDARY_COLOR,
        font=("Segoe UI", 44, "bold")
    ).pack(side="left")

    unit_box = tk.Frame(yield_box, bg=CARD_BG)
    unit_box.pack(side="left", padx=14, pady=8)

    tk.Label(
        unit_box,
        text="Tonnes / Hectare",
        bg=CARD_BG,
        fg=TEXT_COLOR,
        font=("Segoe UI", 14, "bold")
    ).pack(anchor="w")

    tk.Label(
        unit_box,
        text=f"≈ {quintals_ha:.1f} Quintals/Ha   •   {quintals_acre:.1f} Quintals/Acre   •   {kg_val:.0f} kg/Ha",
        bg=CARD_BG,
        fg=MUTED_TEXT,
        font=("Segoe UI", 10)
    ).pack(anchor="w")

    tk.Label(
        yield_box,
        text=badge_text,
        bg=badge_bg,
        fg=badge_fg,
        font=("Segoe UI", 10, "bold"),
        padx=14,
        pady=7
    ).pack(side="right")

    # Grid Info Card
    meta_card = create_card_frame(pred_scroll, title="📍 Prediction Details & Validation", padx=30, pady=10)

    val_grid = tk.Frame(meta_card, bg=CARD_BG)
    val_grid.pack(fill="x")

    left_items = [
        ("Crop Name", crop.title()),
        ("Location", f"{district.title()}, {state.title()}"),
        ("Season", f"{season.title()} Season"),
        ("Model Used", model_name)
    ]

    right_items = [
        (f"Latest Recorded Yield ({prediction['latest_year']})", f"{prediction['actual_yield']:.3f} t/ha"),
        (f"Validation Prediction ({prediction['latest_year']})", f"{prediction['validation_prediction']:.3f} t/ha"),
        ("Prediction Error", f"{prediction['prediction_error']:.3f} t/ha"),
        ("Error Rate", f"{(prediction['prediction_error'] / max(0.001, prediction['actual_yield']) * 100):.1f}%")
    ]

    for row_idx, (lbl, val) in enumerate(left_items):
        tk.Label(val_grid, text=lbl, bg=CARD_BG, fg=MUTED_TEXT, font=("Segoe UI", 10, "bold")).grid(row=row_idx, column=0, sticky="w", padx=10, pady=4)
        tk.Label(val_grid, text=val, bg=CARD_BG, fg=TEXT_COLOR, font=("Segoe UI", 10)).grid(row=row_idx, column=1, sticky="w", padx=10, pady=4)

    for row_idx, (lbl, val) in enumerate(right_items):
        tk.Label(val_grid, text=lbl, bg=CARD_BG, fg=MUTED_TEXT, font=("Segoe UI", 10, "bold")).grid(row=row_idx, column=2, sticky="w", padx=(30, 10), pady=4)
        tk.Label(val_grid, text=val, bg=CARD_BG, fg=ACCENT_BLUE if "Yield" in lbl else TEXT_COLOR, font=("Segoe UI", 10, "bold" if "Yield" in lbl else "normal")).grid(row=row_idx, column=3, sticky="w", padx=10, pady=4)

    # Model Performance Evaluation Metrics
    metric_card = create_card_frame(pred_scroll, title="📊 Model Performance", padx=30, pady=10)

    metrics_container = tk.Frame(metric_card, bg=CARD_BG)
    metrics_container.pack(fill="x", pady=6)

    def add_metric_pill(parent, title, value, subtext, color):
        box = tk.Frame(parent, bg="#F9FAF9", bd=1, relief="solid", highlightbackground=CARD_BORDER, padx=16, pady=12)
        box.pack(side="left", expand=True, fill="both", padx=6)

        tk.Label(box, text=title, bg="#F9FAF9", fg=MUTED_TEXT, font=("Segoe UI", 9, "bold")).pack(anchor="w")
        tk.Label(box, text=value, bg="#F9FAF9", fg=color, font=("Segoe UI", 20, "bold")).pack(anchor="w", pady=(2, 0))
        tk.Label(box, text=subtext, bg="#F9FAF9", fg=MUTED_TEXT, font=("Segoe UI", 8)).pack(anchor="w")

    add_metric_pill(metrics_container, "R² SCORE", f"{performance['r2']:.3f}", "Goodness of fit (0 to 1)", SECONDARY_COLOR)
    add_metric_pill(metrics_container, "RMSE", f"{performance['rmse']:.3f}", "Root mean squared error", ACCENT_BLUE)
    add_metric_pill(metrics_container, "MAE", f"{performance['mae']:.3f}", "Mean absolute error", ACCENT_AMBER)
    add_metric_pill(metrics_container, "DATA SPLIT", f"{summary['training_samples']} / {summary['testing_samples']}", "Train / Test sample count", PRIMARY_COLOR)

    # ==============================================================
    # TAB 2: VISUAL ANALYTICS (STATISTICAL DASHBOARD - NO SOIL GRAPHS)
    # ==============================================================
    graphs_scroll = create_scrollable_container(tab_graphs)

    # Explanatory Guide Banner above the charts
    guide_frame = tk.Frame(graphs_scroll, bg="#EBF3EB", bd=1, relief="solid", highlightbackground="#C8DBC8", padx=16, pady=10)
    guide_frame.pack(fill="x", padx=25, pady=(12, 0))

    tk.Label(
        guide_frame,
        text="📊 Guide to Charts:  (1) Yield Over Time tracks historical harvests and upcoming forecast.  "
             "(2) Distribution benchmarks this prediction against typical yields for this crop.  "
             "(3) Prediction Residuals show historical model error margins (Actual − Fitted).  "
             "(4) Categories show the proportion of past seasons achieving High, Moderate, or Low yields.",
        bg="#EBF3EB",
        fg=PRIMARY_COLOR,
        font=("Segoe UI", 9),
        wraplength=1050,
        justify="left"
    ).pack(anchor="w")

    graph_wrapper = tk.Frame(graphs_scroll, bg=BG_COLOR)
    graph_wrapper.pack(fill="both", expand=True, padx=25, pady=12)

    fig = Figure(figsize=(11.5, 9.2), dpi=100, facecolor="#FFFFFF")
    active_figures.append(fig)
    axs = fig.subplots(2, 2)
    fig.subplots_adjust(hspace=0.40, wspace=0.28, left=0.08, right=0.96, top=0.93, bottom=0.08)

    # -------------------------------------------------------------
    # 1. Line Chart: Historical Actual vs Predicted Yield Trajectory
    # -------------------------------------------------------------
    ax1 = axs[0, 0]
    ax1.set_facecolor("#FAFCFA")
    years = history["Year"].values
    actuals = history["Yield"].values
    preds = history["Predicted"].values

    ax1.plot(years, actuals, marker="o", markersize=6, color="#2E7D32", linewidth=2.2, label="Actual Yield")
    ax1.plot(years, preds, marker="s", markersize=5, color="#1565C0", linestyle="--", linewidth=1.8, label="Model Fitted")

    # Future forecast point
    next_yr = int(years[-1]) + 1
    ax1.plot([years[-1], next_yr], [actuals[-1], pred_val], color="#E65100", linestyle=":", linewidth=1.8)
    ax1.plot(next_yr, pred_val, marker="*", markersize=14, color="#E65100", label=f"Forecast ({next_yr})")

    # Callout on forecast star
    ax1.annotate(
        f"Forecast\n{pred_val:.2f} t/ha",
        xy=(next_yr, pred_val),
        xytext=(0, 12),
        textcoords="offset points",
        ha="center",
        fontsize=8,
        fontweight="bold",
        color="#E65100",
        bbox=dict(boxstyle="round,pad=0.25", fc="#FFF3E0", ec="#E65100", lw=1)
    )

    ax1.set_xlim(years[0] - 0.4, next_yr + 0.6)
    ax1.set_title(f"Yield Over Time: {crop.title()} ({district.title()})", fontsize=11, fontweight="bold", color=PRIMARY_COLOR)
    ax1.set_xlabel("Harvest Year", fontsize=9, color=TEXT_COLOR)
    ax1.set_ylabel("Yield (Tonnes / Ha)", fontsize=9, color=TEXT_COLOR)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=8, loc="best", framealpha=0.9)

    # -------------------------------------------------------------
    # 2. Histogram & Density Curve: Regional Crop Yield Distribution
    # -------------------------------------------------------------
    ax2 = axs[0, 1]
    ax2.set_facecolor("#FAFCFA")

    sample_yields = crop_yield_series if len(crop_yield_series) > 10 else actuals
    n_bins = min(22, max(8, int(len(sample_yields) ** 0.5)))

    counts, bins, _ = ax2.hist(
        sample_yields,
        bins=n_bins,
        color="#81C784",
        edgecolor="#2E7D32",
        alpha=0.65,
        density=True,
        label="Yield Frequency"
    )

    mu = float(np.mean(sample_yields))
    sigma = max(0.01, float(np.std(sample_yields)))
    x_axis = np.linspace(min(sample_yields), max(sample_yields), 100)
    pdf = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_axis - mu) / sigma) ** 2)
    ax2.plot(x_axis, pdf, color="#1B5E20", linewidth=1.8, label="Normal Curve")

    ax2.axvline(pred_val, color="#D32F2F", linestyle="--", linewidth=2.0, label=f"Forecast: {pred_val:.2f} t/ha")
    ax2.axvline(mu, color="#1565C0", linestyle=":", linewidth=1.5, label=f"Mean: {mu:.2f} t/ha")

    diff_pct = ((pred_val - mu) / mu) * 100 if mu > 0 else 0
    ax2.text(
        0.04, 0.90,
        f"Benchmark: {'+' if diff_pct >= 0 else ''}{diff_pct:.1f}% vs Regional Mean",
        transform=ax2.transAxes,
        fontsize=8,
        fontweight="bold",
        color=PRIMARY_COLOR,
        bbox=dict(boxstyle="round,pad=0.25", fc="#E8F5E9", ec="#A5D6A7", lw=1)
    )

    ax2.set_title(f"Yield Distribution & Benchmark ({crop.title()})", fontsize=11, fontweight="bold", color=PRIMARY_COLOR)
    ax2.set_xlabel("Yield (Tonnes / Ha)", fontsize=9, color=TEXT_COLOR)
    ax2.set_ylabel("Probability Density", fontsize=9, color=TEXT_COLOR)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=8, loc="upper right", framealpha=0.9)

    # -------------------------------------------------------------
    # 3. Bar Chart: Model Residual Calibration Errors (Actual - Predicted)
    # -------------------------------------------------------------
    ax3 = axs[1, 0]
    ax3.set_facecolor("#FAFCFA")
    residuals = history["Residual"].values

    bar_colors = ["#2E7D32" if r >= 0 else "#C62828" for r in residuals]
    bars = ax3.bar(years, residuals, color=bar_colors, width=0.45, edgecolor="#333333", alpha=0.85)
    ax3.axhline(0, color="#1C251D", linestyle="-", linewidth=1.2)

    for bar, val in zip(bars, residuals):
        y_offset = 0.015 if val >= 0 else -0.035
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_offset,
            f"{val:+.2f}",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=8,
            fontweight="bold",
            color=TEXT_COLOR
        )

    y_min, y_max = min(residuals), max(residuals)
    margin = max(0.12, max(abs(y_min), abs(y_max)) * 0.35)
    ax3.set_ylim(min(y_min - margin, -0.12), max(y_max + margin, 0.12))

    ax3.set_title("Historical Error Margins (Actual − Predicted)", fontsize=11, fontweight="bold", color=PRIMARY_COLOR)
    ax3.set_xlabel("Harvest Year", fontsize=9, color=TEXT_COLOR)
    ax3.set_ylabel("Residual (t/ha)", fontsize=9, color=TEXT_COLOR)
    ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax3.grid(True, linestyle=":", alpha=0.6)

    # -------------------------------------------------------------
    # 4. Pie Diagram: Productivity Tier Breakdown (Modern Donut)
    # -------------------------------------------------------------
    ax4 = axs[1, 1]
    ax4.set_facecolor("#FAFCFA")

    tier_high = int((sample_yields >= 2.5).sum())
    tier_moderate = int(((sample_yields >= 1.5) & (sample_yields < 2.5)).sum())
    tier_low = int((sample_yields < 1.5).sum())

    tier_sizes = [tier_high, tier_moderate, tier_low]
    tier_labels = ["High (≥2.5 t/ha)", "Moderate (1.5–2.5)", "Low (<1.5 t/ha)"]
    tier_colors = ["#2E7D32", "#FFA000", "#D32F2F"]

    filtered_sizes = []
    filtered_labels = []
    filtered_colors = []
    for s, l, c in zip(tier_sizes, tier_labels, tier_colors):
        if s > 0:
            filtered_sizes.append(s)
            filtered_labels.append(l)
            filtered_colors.append(c)

    if filtered_sizes:
        wedges, texts, autotexts = ax4.pie(
            filtered_sizes,
            labels=filtered_labels,
            colors=filtered_colors,
            autopct="%1.1f%%",
            startangle=140,
            pctdistance=0.75,
            textprops={"fontsize": 8, "color": TEXT_COLOR}
        )
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")
            autotext.set_fontsize(8)

        centre_circle = Circle((0, 0), 0.52, fc="white")
        ax4.add_artist(centre_circle)
        ax4.text(0, 0.05, f"{len(sample_yields)}", ha="center", va="center", fontsize=14, fontweight="bold", color=PRIMARY_COLOR)
        ax4.text(0, -0.15, "Past Records", ha="center", va="center", fontsize=8, color=MUTED_TEXT)
        ax4.axis("equal")
    else:
        ax4.text(0.5, 0.5, "Data Unavailable", ha="center", va="center")

    ax4.set_title(f"Productivity Tier Breakdown ({crop.title()})", fontsize=11, fontweight="bold", color=PRIMARY_COLOR)

    # Render Matplotlib Canvas
    canvas = FigureCanvasTkAgg(fig, master=graph_wrapper)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

    graph_wrapper.update_idletasks()
    if hasattr(graphs_scroll, "_parent_canvas"):
        graphs_scroll._parent_canvas.configure(scrollregion=graphs_scroll._parent_canvas.bbox("all"))

    # ==============================================================
    # TAB 3: HISTORICAL RECORDS
    # ==============================================================
    hist_frame = tk.Frame(tab_history, bg=BG_COLOR, padx=25, pady=16)
    hist_frame.pack(fill="both", expand=True)

    tk.Label(
        hist_frame,
        text=f"Historical Yield Records: {district.title()}, {state.title()}",
        bg=BG_COLOR,
        fg=PRIMARY_COLOR,
        font=("Segoe UI", 12, "bold")
    ).pack(anchor="w", pady=(0, 10))

    create_styled_tree(hist_frame, history)

    # ==============================================================
    # TAB 4: WEATHER & SOIL DATA
    # ==============================================================
    weather_frame = tk.Frame(tab_weather, bg=BG_COLOR, padx=25, pady=16)
    weather_frame.pack(fill="both", expand=True)

    tk.Label(
        weather_frame,
        text="Weather & Soil Readings Over Time",
        bg=BG_COLOR,
        fg=PRIMARY_COLOR,
        font=("Segoe UI", 12, "bold")
    ).pack(anchor="w", pady=(0, 10))

    create_styled_tree(weather_frame, weather)

    # ==============================================================
    # TAB 5: REPRESENTATIVE FEATURES
    # ==============================================================
    feat_frame = tk.Frame(tab_features, bg=BG_COLOR, padx=25, pady=16)
    feat_frame.pack(fill="both", expand=True)

    tk.Label(
        feat_frame,
        text="Values Used for This Prediction",
        bg=BG_COLOR,
        fg=PRIMARY_COLOR,
        font=("Segoe UI", 12, "bold")
    ).pack(anchor="w", pady=(0, 10))

    create_styled_tree(feat_frame, features)

    # ==============================================================
    # TAB 6: SUMMARY
    # ==============================================================
    summary_scroll = create_scrollable_container(tab_summary)

    scope_card = create_card_frame(summary_scroll, title="📁 Dataset Scope", padx=30, pady=12)
    s_grid = tk.Frame(scope_card, bg=CARD_BG)
    s_grid.pack(fill="x")

    scope_rows = [
        ("Total Records in Dataset", f"{summary['total_records']:,}"),
        (f"Records for {crop.title()}", f"{summary['crop_records']:,}"),
        (f"Records for {district.title()}", f"{summary['historical_records']}"),
        ("Training Samples (80%)", f"{summary['training_samples']:,}"),
        ("Testing Samples (20%)", f"{summary['testing_samples']:,}")
    ]

    for idx, (label, val) in enumerate(scope_rows):
        tk.Label(s_grid, text=label, bg=CARD_BG, fg=MUTED_TEXT, font=("Segoe UI", 10)).grid(row=idx, column=0, sticky="w", pady=4)
        tk.Label(s_grid, text=val, bg=CARD_BG, fg=PRIMARY_COLOR, font=("Segoe UI", 10, "bold")).grid(row=idx, column=1, sticky="w", padx=30, pady=4)

    stat_card = create_card_frame(summary_scroll, title="📈 Yield Statistics", padx=30, pady=12)
    stat_grid = tk.Frame(stat_card, bg=CARD_BG)
    stat_grid.pack(fill="x")

    dist_rows = [
        ("Minimum Yield in Dataset", f"{summary['yield_min']:.3f} t/ha"),
        ("Maximum Yield in Dataset", f"{summary['yield_max']:.3f} t/ha"),
        ("Average Yield in Dataset", f"{summary['yield_mean']:.3f} t/ha"),
        ("District 5-Year Average", f"{summary['historical_mean']:.3f} t/ha"),
        ("District 5-Year Median", f"{summary['historical_median']:.3f} t/ha"),
        ("District Standard Deviation", f"{summary['historical_std']:.3f} t/ha")
    ]

    for idx, (label, val) in enumerate(dist_rows):
        tk.Label(stat_grid, text=label, bg=CARD_BG, fg=MUTED_TEXT, font=("Segoe UI", 10)).grid(row=idx, column=0, sticky="w", pady=4)
        tk.Label(stat_grid, text=val, bg=CARD_BG, fg=SECONDARY_COLOR, font=("Segoe UI", 10, "bold")).grid(row=idx, column=1, sticky="w", padx=30, pady=4)

    # Attach universal mousewheel and keyboard scrolling
    setup_window_scrolling(window, notebook)

    return window
