import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from crop_yield_prediction import detailed_prediction


# ======================================================
# TREEVIEW FUNCTION
# ======================================================

def create_tree(parent, dataframe):

    tree_frame = tk.Frame(parent)
    tree_frame.pack(fill="both", expand=True)

    style = ttk.Style()
    style.theme_use("clam")

    style.configure("Treeview.Heading",
                    background="#2E7D32",
                    foreground="white",
                    font=("Segoe UI", 11, "bold"),
                    padding=8)

    style.configure("Treeview",
                    background="white",
                    foreground="#333333",
                    rowheight=35,
                    font=("Segoe UI",10),
                    fieldbackground="white")

    style.map("Treeview",
              background=[("selected","#A5D6A7")],
              foreground=[("selected","black")])

    tree = ttk.Treeview(tree_frame)
    tree["columns"] = list(dataframe.columns)
    tree["show"] = "headings"

    for col in dataframe.columns:
        tree.heading(col, text=col)
        tree.column(col, width=180, anchor="center")   # ⭐ wider columns

    for index, row in dataframe.iterrows():
        # ⭐ round long floats for cleaner UI
        values = [round(v, 3) if isinstance(v, float) else v for v in row]

        tag = "evenrow" if index % 2 == 0 else "oddrow"
        tree.insert("", "end", values=values, tags=(tag,))

    tree.tag_configure("evenrow", background="#F1F8E9")
    tree.tag_configure("oddrow", background="white")

    scroll_y = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
    scroll_x = ttk.Scrollbar(tree_frame, orient="horizontal", command=tree.xview)

    tree.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)

    tree.grid(row=0, column=0, sticky="nsew")
    scroll_y.grid(row=0, column=1, sticky="ns")
    scroll_x.grid(row=1, column=0, sticky="ew")

    tree_frame.grid_rowconfigure(0, weight=1)
    tree_frame.grid_columnconfigure(0, weight=1)
    def _on_shift_mousewheel(event):
        tree.xview_scroll(int(-1 * (event.delta / 120)), "units")

    tree.bind("<Shift-MouseWheel>", _on_shift_mousewheel)
    return tree



# ======================================================
# CARD FUNCTION
# ======================================================

def create_card(parent):
    card = tk.Frame(parent, bg="white", bd=2, relief="groove")
    card.pack(fill="x", padx=30, pady=10)
    return card


# ======================================================
# SCROLLABLE CONTENT
# ======================================================

def create_scrollable_content(parent):

    canvas = tk.Canvas(parent, bg="#F4F8F4", highlightthickness=0)
    scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)

    scroll_frame = tk.Frame(canvas, bg="#F4F8F4")

    scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    window_id = canvas.create_window((0,0), window=scroll_frame, anchor="nw")

    canvas.bind("<Configure>", lambda e: canvas.itemconfig(window_id, width=e.width))
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
    canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

    return scroll_frame


# ======================================================
# SHOW WINDOW
# ======================================================

def show_window(state, district, crop, season, model_type="rf"):

    result = detailed_prediction(state, district, crop, season, model_type=model_type)

    model_display_name = result["prediction"].get("model_type", "Machine Learning")

    window = tk.Toplevel()
    window.title(f"Detailed Crop Yield Report ({model_display_name})")
    window.geometry("1300x720")
    window.configure(bg="#F4F8F4")
    window.resizable(True, True)

    header = tk.Frame(window, bg="#1B5E20", height=80)
    header.pack(fill="x")

    tk.Label(
        header,
        text=f"🌾 DETAILED CROP YIELD REPORT ({model_display_name.upper()})",
        bg="#1B5E20",
        fg="white",
        font=("Segoe UI",22,"bold")
    ).pack(pady=18)

    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TNotebook", background="#F4F8F4")
    style.configure("TNotebook.Tab", font=("Segoe UI",11,"bold"), padding=(20,10))

    notebook = ttk.Notebook(window)
    notebook.pack(fill="both", expand=True, padx=20, pady=20)

    prediction_tab = tk.Frame(notebook)
    history_tab = tk.Frame(notebook)
    weather_tab = tk.Frame(notebook)
    features_tab = tk.Frame(notebook)
    summary_tab = tk.Frame(notebook)

    notebook.add(prediction_tab, text="Prediction")
    notebook.add(history_tab, text="History")
    notebook.add(weather_tab, text="Weather & Soil")
    notebook.add(features_tab, text="Features")
    notebook.add(summary_tab, text="Summary")

    def next_tab():
        current = notebook.index(notebook.select())
        total = notebook.index("end")
        if current < total - 1:
            notebook.select(current + 1)

    def previous_tab():
        current = notebook.index(notebook.select())
        if current > 0:
            notebook.select(current - 1)

    def hover(button, normal, hover_color):
        button.bind("<Enter>", lambda e: button.config(bg=hover_color))
        button.bind("<Leave>", lambda e: button.config(bg=normal))

    def create_tab_layout(tab, title):
        container = tk.Frame(tab, bg="#F4F8F4")
        container.pack(fill="both", expand=True)

        tk.Label(
            container,
            text=title,
            bg="#F4F8F4",
            fg="#1B5E20",
            font=("Segoe UI",20,"bold")
        ).pack(pady=(15,10))

        content_container = tk.Frame(container, bg="#F4F8F4")
        content_container.pack(fill="both", expand=True)

        content = create_scrollable_content(content_container)

        nav = tk.Frame(container, bg="#F4F8F4")
        nav.pack(fill="x", pady=10)

        return content, nav

    def add_navigation(nav):
        center = tk.Frame(nav, bg="#F4F8F4")
        center.pack()

        def on_close():
            plt.close("all")
            window.destroy()

        window.protocol("WM_DELETE_WINDOW", on_close)

        prev = tk.Button(center, text="◀ Previous", bg="#FB8C00", fg="white",
                         font=("Segoe UI",10,"bold"), width=14, relief="flat",
                         command=previous_tab)

        home = tk.Button(center, text="🏠 Home", bg="#C62828", fg="white",
                         font=("Segoe UI",10,"bold"), width=14, relief="flat",
                         command=on_close)

        nxt = tk.Button(center, text="Next ▶", bg="#2E7D32", fg="white",
                        font=("Segoe UI",10,"bold"), width=14, relief="flat",
                        command=next_tab)

        prev.grid(row=0, column=0, padx=15, pady=10)
        home.grid(row=0, column=1, padx=15, pady=10)
        nxt.grid(row=0, column=2, padx=15, pady=10)

        hover(prev, "#FB8C00", "#EF6C00")
        hover(home, "#C62828", "#B71C1C")
        hover(nxt, "#2E7D32", "#1B5E20")

    prediction = result["prediction"]
    performance = result["performance"]

    # ======================================================
    # PREDICTION TAB
    # ======================================================

    prediction_content, prediction_nav = create_tab_layout(prediction_tab, "🌾 Prediction Overview")

    info_card = create_card(prediction_content)

    fields = [
        ("📍 State", prediction["state"]),
        ("🏙 District", prediction["district"]),
        ("🌾 Crop", prediction["crop"]),
        ("☀ Season", prediction["season"]),
        ("🤖 Model Used", prediction.get("model_type", "Machine Learning"))
    ]

    for i, (label, value) in enumerate(fields):
        tk.Label(info_card, text=label, bg="white", fg="#1B5E20",
                 font=("Segoe UI",12,"bold")).grid(row=i, column=0, padx=20, pady=10, sticky="w")
        tk.Label(info_card, text=value, bg="white",
                 font=("Segoe UI",12)).grid(row=i, column=1, padx=20, sticky="w")

    yield_card = create_card(prediction_content)

    tk.Label(yield_card, text="🌾 Predicted Yield", bg="#E8F5E9",
             fg="#1B5E20", font=("Segoe UI",13,"bold")).pack(pady=(10,5))

    tk.Label(yield_card, text=f"{prediction['predicted_yield']:.3f} Tonnes/Hectare",
             bg="#E8F5E9", fg="#2E7D32", font=("Segoe UI",24,"bold")).pack(pady=(0,15))

    metric_card = create_card(prediction_content)

    metrics = [
        ("📈 Actual Yield", prediction["actual_yield"]),
        ("🎯 Validation Prediction", prediction["validation_prediction"]),
        ("❗ Prediction Error", prediction["prediction_error"]),
        ("📉 MAE", performance["mae"]),
        ("📊 RMSE", performance["rmse"]),
        ("✅ R² Score", performance["r2"])
    ]

    for i, (label, value) in enumerate(metrics):
        tk.Label(metric_card, text=label, bg="white",
                 font=("Segoe UI",11,"bold")).grid(row=i, column=0, padx=20, pady=8, sticky="w")
        tk.Label(metric_card, text=f"{value:.3f}", bg="white", fg="#1565C0",
                 font=("Segoe UI",11)).grid(row=i, column=1, padx=20, sticky="w")

    add_navigation(prediction_nav)

    # ======================================================
    # HISTORY TAB
    # ======================================================

    history_content, history_nav = create_tab_layout(history_tab, "📈 Historical Crop Yield Records")

    history_frame = tk.Frame(history_content)
    history_frame.pack(fill="both", expand=True, padx=10, pady=10)

    create_tree(history_frame, result["history"])
    add_navigation(history_nav)

    # ======================================================
    # WEATHER TAB
    # ======================================================

    weather_content, weather_nav = create_tab_layout(weather_tab, "☁ Weather & Soil Data")

    weather_frame = tk.Frame(weather_content,height=420)
    weather_frame.pack(fill="x", expand=True, padx=10, pady=10)
    weather_frame.pack_propagate(False)

    weather_tree=create_tree(weather_frame, result["weather"])

    for col in weather_tree["columns"]:
        weather_tree.column(col, stretch=False)

    weather_tree.column("Year", width=90)

    weather_tree.column("Rainfall", width=170)  

    weather_tree.column("Temperature", width=180)
    
    weather_tree.column("Nitrogen", width=170)

    weather_tree.column("Phosphorus", width=170)

    weather_tree.column("Potassium", width=170)

    weather_tree.column("Soil pH", width=150)

    weather_tree.column("Organic Carbon", width=220)

    weather_tree.column("Zinc", width=170)

    weather_tree.column("Iron", width=150)
    weather_tree.column("Copper", width=150)
    weather_tree.column("Boron", width=150)
    weather_tree.column("Manganese", width=150)
    weather_tree.column("Sulphur", width=150)
    weather_tree.column("Yield", width=150)
    
    
    add_navigation(weather_nav)

    # ======================================================
    # FEATURES TAB
    # ======================================================

    features_content, features_nav = create_tab_layout(features_tab, "🧠 Features Used For Prediction")

    feature_frame = tk.Frame(features_content)
    feature_frame.pack(fill="both", expand=True, padx=10, pady=10)

    create_tree(feature_frame, result["features"])
    add_navigation(features_nav)

    # ======================================================
    # SUMMARY TAB
    # ======================================================

    summary = result["summary"]

    summary_content, summary_nav = create_tab_layout(summary_tab, "📋 Dataset Summary")

    summary_frame = tk.Frame(summary_content)
    summary_frame.pack(fill="both", expand=True, padx=20, pady=20)

    dataset_card = create_card(summary_content)

    dataset_info = [
        ("📁 Total Dataset Records", summary["total_records"]),
        ("🌾 Crop Records Used", summary["crop_records"]),
        ("📜 Historical Records", summary["historical_records"]),
        ("🧠 Training Samples", summary["training_samples"]),
        ("🧪 Testing Samples", summary["testing_samples"])
    ]

    for i, (label, value) in enumerate(dataset_info):
        tk.Label(dataset_card, text=label, bg="white", fg="#1B5E20",
                 font=("Segoe UI",11,"bold")).grid(row=i, column=0, padx=20, pady=8, sticky="w")
        tk.Label(dataset_card, text=value, bg="white", fg="#1565C0",
                 font=("Segoe UI",11)).grid(row=i, column=1, padx=20, sticky="w")

    yield_card = create_card(summary_content)

    tk.Label(yield_card, text="📈 Dataset Yield Statistics", bg="#E8F5E9",
             fg="#1B5E20", font=("Segoe UI",13,"bold")).pack(pady=(10,5))

    stats = [
        ("Minimum Yield", summary["yield_min"]),
        ("Maximum Yield", summary["yield_max"]),
        ("Average Yield", summary["yield_mean"])
    ]

    for text, value in stats:
        row = tk.Frame(yield_card, bg="#E8F5E9")
        row.pack(fill="x", padx=25, pady=4)

        tk.Label(row, text=text, bg="#E8F5E9",
                 font=("Segoe UI",11,"bold")).pack(side="left")
        tk.Label(row, text=f"{value:.3f}", bg="#E8F5E9",
                 fg="#2E7D32", font=("Segoe UI",11)).pack(side="right")

    history_card = create_card(summary_content)

    history_data = [
        ("📊 Mean Yield", summary["historical_mean"]),
        ("📉 Median Yield", summary["historical_median"])
    ]

    for i, (label, value) in enumerate(history_data):
        tk.Label(history_card, text=label, bg="white", fg="#1B5E20",
                 font=("Segoe UI",11,"bold")).grid(row=i, column=0, padx=20, pady=8, sticky="w")
        tk.Label(history_card, text=f"{value:.3f}", bg="white",
                 fg="#1565C0", font=("Segoe UI",11)).grid(row=i, column=1, padx=20, sticky="w")

    add_navigation(summary_nav)

    # ======================================================
    # 📊 GRAPHS TAB (NO FEATURE IMPORTANCE, NO SOIL NUTRIENTS)
    # ======================================================

    sns.set_style("whitegrid")

    graphs_tab = tk.Frame(notebook)
    notebook.add(graphs_tab, text="📊 Graphs")

    graphs_content, graphs_nav = create_tab_layout(graphs_tab, "📊 Regression Graphs")

    graph_frame = tk.Frame(graphs_content, bg="#F4F8F4")
    graph_frame.pack(fill="both", expand=True, padx=10, pady=10)

    def add_graph(fig, parent):
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.pack(fill="both", expand=True, pady=20)

    # Yield Trend
    fig1, ax1 = plt.subplots(figsize=(7,4))
    ax1.plot(result["history"]["Year"], result["history"]["Yield"], marker="o", color="#2E7D32")
    ax1.set_title("Yield Trend Over Years", fontsize=14)
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Yield (Tonnes/Hectare)")
    add_graph(fig1, graph_frame)

    # Rainfall vs Yield
    fig2, ax2 = plt.subplots(figsize=(7,4))
    ax2.scatter(result["weather"]["Rainfall"], result["weather"]["Yield"], color="#1565C0")
    ax2.set_title("Rainfall vs Yield", fontsize=14)
    ax2.set_xlabel("Rainfall (mm)")
    ax2.set_ylabel("Yield (Tonnes/Hectare)")
    add_graph(fig2, graph_frame)

    # Residual Plot (Simple)
    actual = result["history"]["Yield"]
    predicted = [prediction["predicted_yield"]] * len(actual)
    residuals = actual - predicted

    fig3, ax3 = plt.subplots(figsize=(7,4))
    ax3.scatter(predicted, residuals, color="#8E24AA")
    ax3.axhline(0, color="black", linestyle="--")
    ax3.set_title("Residual Plot", fontsize=14)
    ax3.set_xlabel("Predicted Yield")
    ax3.set_ylabel("Residual (Actual - Predicted)")
    add_graph(fig3, graph_frame)

    add_navigation(graphs_nav)
