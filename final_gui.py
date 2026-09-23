import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import os
from crop_yield_prediction import predict_crop

# =====================================================
# LOAD DATASET
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "final_merged_and_cleaned_dataset_3.csv")

if not os.path.exists(CSV_FILE):
    messagebox.showerror(
        "File Not Found",
        f"Cannot find {CSV_FILE}\n\nPlace the CSV in the same folder."
    )
    raise SystemExit

df = pd.read_csv(CSV_FILE)

df.columns = df.columns.str.strip()

for col in ["state_name", "district_name", "crop_name", "season"]:
    df[col] = df[col].astype(str).str.strip()

# =====================================================
# MAIN WINDOW
# =====================================================

root = tk.Tk()

root.title("AI Crop Yield Prediction System")

root.state("zoomed")

root.resizable(False, False)

# =====================================================
# COLORS
# =====================================================

PRIMARY = "#1B5E20"
SECONDARY = "#2E7D32"
BACKGROUND = "#EEF6EE"
CARD = "#FFFFFF"
ACCENT = "#1565C0"
WARNING = "#FB8C00"
DANGER = "#C62828"
TEXT = "#333333"

root.configure(bg=BACKGROUND)

# =====================================================
# STYLE
# =====================================================

style = ttk.Style()

style.theme_use("clam")

style.configure(
    "TCombobox",
    padding=6,
    font=("Segoe UI", 11)
)

style.configure(
    "Horizontal.TProgressbar",
    thickness=12
)

# =====================================================
# HEADER
# =====================================================

header = tk.Frame(
    root,
    bg=PRIMARY,
    height=70
)

header.pack(fill="x")

header.pack_propagate(False)

tk.Label(
    header,
    text="🌾 CROP YIELD PREDICTION SYSTEM",
    bg=PRIMARY,
    fg="white",
    font=("Segoe UI", 24, "bold")
).pack(pady=(14, 0))
# =====================================================
# INFORMATION STRIP
# =====================================================

info_bar = tk.Frame(
    root,
    bg="#E8F5E9",
    height=30
)

info_bar.pack(fill="x")

info_bar.pack_propagate(False)

tk.Label(
    info_bar,
    text=(
        f"Dataset Records : {len(df):,}     "
        f"States : {df['state_name'].nunique()}     "
        f"Districts : {df['district_name'].nunique()}     "
        f"Crops : {df['crop_name'].nunique()}     "
        f"Models : Random Forest & Linear Regression"
    ),
    bg="#E8F5E9",
    fg=PRIMARY,
    font=("Segoe UI", 10, "bold")
).pack(pady=9)
# =====================================================
# MAIN CONTAINER
# =====================================================

main_container = tk.Frame(
    root,
    bg=BACKGROUND
)

main_container.pack(
    fill="both",
    expand=True,
    padx=20,
    pady=15
)

# =====================================================
# PARAMETER CARD (TOP)
# =====================================================

input_card = tk.Frame(
    main_container,
    bg=CARD,
    bd=2,
    relief="groove"
)

input_card.pack(
    fill="x"
)

tk.Label(
    input_card,
    text="📋 Prediction Parameters",
    bg=CARD,
    fg=PRIMARY,
    font=("Segoe UI",16,"bold")
).pack(
    pady=(0,0)
)

form = tk.Frame(
    input_card,
    bg=CARD
)

form.pack(
    pady=(0,0)
)

# =====================================================
# PROGRESS ROW
# =====================================================

style.configure(
    "Green.Horizontal.TProgressbar",
    thickness=6     # Reduce thickness (default was 12)
)

progress_card = tk.Frame(
    main_container,
    bg=CARD,
    bd=2,
    relief="groove"
)

progress_card.pack(
    fill="x",
    pady=(2,2)
)

tk.Label(
    progress_card,
    text="⚙ Prediction Progress",
    bg=CARD,
    fg=PRIMARY,
    font=("Segoe UI",12,"bold")
).pack(
    anchor="w",
    padx=15,
    pady=(2,4)
)

progress_status = tk.Label(
    progress_card,
    text="Ready for Prediction",
    bg=CARD,
    fg="#666666",
    font=("Segoe UI",9)
)

progress_status.pack(
    anchor="w",
    padx=10,
    pady=0
)

progress = ttk.Progressbar(
    progress_card,
    style="Green.Horizontal.TProgressbar",
    orient="horizontal",
    mode="determinate"
)

progress.pack(
    fill="x",
    padx=15,
    pady=(2,4)
)
# =====================================================
# RESULT AREA
# =====================================================

result_frame = tk.Frame(
    main_container,
    bg=BACKGROUND
)

result_frame.pack(
    fill="both",
    expand=True
)


# =====================================================
# PREDICTION CARD
# =====================================================

prediction_card = tk.Frame(
    result_frame,
    bg=CARD,
    bd=2,
    relief="groove",
    width=560,
    height=120
)

prediction_card.pack(
    side="left",
    fill="both",
    expand=True,
    padx=(0,10)
)

prediction_card.pack_propagate(False)


# =====================================================
# PERFORMANCE CARD
# =====================================================

performance_card = tk.Frame(
    result_frame,
    bg=CARD,
    bd=2,
    relief="groove",
    width=560,
    height=120
)

performance_card.pack(
    side="left",
    fill="both",
    expand=True
)

performance_card.pack_propagate(False)


# =====================================================
# BUTTON AREA
# =====================================================

button_bar = tk.Frame(
    main_container,
    bg=BACKGROUND
)

button_bar.pack(
    fill="x",
    pady=(18,10)
)
# =====================================================
# VARIABLES
# =====================================================

state_var = tk.StringVar()

district_var = tk.StringVar()

crop_var = tk.StringVar()

season_var = tk.StringVar()

states = sorted(
    df["state_name"].dropna().unique()
)
# =====================================================
# UPDATE COMBOBOX FUNCTIONS
# =====================================================

def update_districts(event=None):

    selected_state = state_var.get()

    districts = sorted(
        df[df["state_name"] == selected_state]["district_name"]
        .dropna()
        .unique()
    )

    district_combo["values"] = districts

    district_var.set("")
    crop_var.set("")
    season_var.set("")

    crop_combo["values"] = []
    season_combo["values"] = []


def update_crops(event=None):

    selected_state = state_var.get()
    selected_district = district_var.get()

    crops = sorted(
        df[
            (df["state_name"] == selected_state)
            &
            (df["district_name"] == selected_district)
        ]["crop_name"]
        .dropna()
        .unique()
    )

    crop_combo["values"] = crops

    crop_var.set("")
    season_var.set("")

    season_combo["values"] = []


def update_seasons(event=None):

    selected_state = state_var.get()
    selected_district = district_var.get()
    selected_crop = crop_var.get()

    seasons = sorted(
        df[
            (df["state_name"] == selected_state)
            &
            (df["district_name"] == selected_district)
            &
            (df["crop_name"] == selected_crop)
        ]["season"]
        .dropna()
        .unique()
    )

    season_combo["values"] = seasons

    season_var.set("")

# =====================================================
# HORIZONTAL PARAMETER BAR
# =====================================================

parameter_frame = tk.Frame(
    form,
    bg=CARD
)

parameter_frame.pack(
    padx=20,
    pady=2
)

label_font = ("Segoe UI",10,"bold")

combo_width = 20

# ---------------- State ----------------

tk.Label(
    parameter_frame,
    text="📍 State",
    bg=CARD,
    fg=PRIMARY,
    font=label_font
).grid(
    row=0,
    column=0,
    sticky="w",
    padx=10
)

state_combo = ttk.Combobox(
    parameter_frame,
    textvariable=state_var,
    width=combo_width,
    state="readonly"
)

state_combo["values"] = states

state_combo.grid(
    row=1,
    column=0,
    padx=10,
    pady=(2,0)
)

state_combo.bind(
    "<<ComboboxSelected>>",
    update_districts
)

# ---------------- District ----------------

tk.Label(
    parameter_frame,
    text="🏙 District",
    bg=CARD,
    fg=PRIMARY,
    font=label_font
).grid(
    row=0,
    column=1,
    sticky="w",
    padx=10
)

district_combo = ttk.Combobox(
    parameter_frame,
    textvariable=district_var,
    width=combo_width,
    state="readonly"
)

district_combo.grid(
    row=1,
    column=1,
    padx=10,
    pady=(2,0)
)

district_combo.bind(
    "<<ComboboxSelected>>",
    update_crops
)

# ---------------- Crop ----------------

tk.Label(
    parameter_frame,
    text="🌾 Crop",
    bg=CARD,
    fg=PRIMARY,
    font=label_font
).grid(
    row=0,
    column=2,
    sticky="w",
    padx=10
)

crop_combo = ttk.Combobox(
    parameter_frame,
    textvariable=crop_var,
    width=combo_width,
    state="readonly"
)

crop_combo.grid(
    row=1,
    column=2,
    padx=10,
    pady=(2,0)
)

crop_combo.bind(
    "<<ComboboxSelected>>",
    update_seasons
)

# ---------------- Season ----------------

tk.Label(
    parameter_frame,
    text="☀ Season",
    bg=CARD,
    fg=PRIMARY,
    font=label_font
).grid(
    row=0,
    column=3,
    sticky="w",
    padx=10
)

season_combo = ttk.Combobox(
    parameter_frame,
    textvariable=season_var,
    width=combo_width,
    state="readonly"
)

season_combo.grid(
    row=1,
    column=3,
    padx=10,
    pady=(2,0)
)

tk.Label(
    input_card,
    text="💡 Select State → District → Crop → Season",
    bg=CARD,
    fg="#666666",
    font=("Segoe UI",9)
).pack(
    pady=(4,6)
)
# =====================================================
# VALIDATION
# =====================================================

def validate_inputs():

    if state_var.get() == "":
        messagebox.showerror(
            "Missing Input",
            "Please select a State."
        )
        return False

    if district_var.get() == "":
        messagebox.showerror(
            "Missing Input",
            "Please select a District."
        )
        return False

    if crop_var.get() == "":
        messagebox.showerror(
            "Missing Input",
            "Please select a Crop."
        )
        return False

    if season_var.get() == "":
        messagebox.showerror(
            "Missing Input",
            "Please select a Season."
        )
        return False

    return True
# =====================================================
# PREDICTION CARD CONTENT
# =====================================================

tk.Label(
    prediction_card,
    text="🌾 Predicted Crop Yield",
    bg=CARD,
    fg=PRIMARY,
    font=("Segoe UI",18,"bold")
).pack(
    pady=(9,4)
)

yield_label = tk.Label(
    prediction_card,
    text="--",
    bg=CARD,
    fg=SECONDARY,
    font=("Segoe UI",50,"bold")
)

yield_label.pack(pady=(5,0))

tk.Label(
    prediction_card,
    text="Tonnes / Hectare",
    bg=CARD,
    fg="#666666",
    font=("Segoe UI",13)
).pack()

ttk.Separator(
    prediction_card,
    orient="horizontal"
).pack(
    fill="x",
    padx=35,
    pady=10
)

prediction_message = tk.Label(
    prediction_card,
    text="Waiting for prediction...",
    bg=CARD,
    fg="#555555",
    font=("Segoe UI",11),
    justify="center"
)

prediction_message.pack(
    pady=15
)

# =====================================================
# PERFORMANCE CARD CONTENT
# =====================================================

performance_title = tk.Label(
    performance_card,
    text="📊 Model Performance",
    bg=CARD,
    fg=PRIMARY,
    font=("Segoe UI",18,"bold")
)
performance_title.pack(
    pady=(9,4)
)

metric_row = tk.Frame(
    performance_card,
    bg=CARD
)

metric_row.pack(
    fill="both",
    expand=True,
    padx=20
)

def metric_card(parent, title, color):

    card = tk.Frame(
        parent,
        bg="#F8FAF8",
        bd=1,
        relief="solid",
        width=150,
        height=120
    )

    card.pack(
        side="left",
        expand=True,
        padx=8,
        pady=8
    )

    card.pack_propagate(False)

    tk.Label(
        card,
        text=title,
        bg="#F8FAF8",
        fg=color,
        font=("Segoe UI",12,"bold")
    ).pack(
        pady=(35,10)
    )

    value = tk.Label(
        card,
        text="--",
        bg="#F8FAF8",
        fg=TEXT,
        font=("Segoe UI",24,"bold")
    )

    value.pack()

    return value
mae_label = metric_card(
    metric_row,
    "📉 MAE",
    "#EF6C00"
)

rmse_label = metric_card(
    metric_row,
    "📊 RMSE",
    ACCENT
)

r2_label = metric_card(
    metric_row,
    "✅ R²",
    SECONDARY
)
# =====================================================
# PREDICT FUNCTION
# =====================================================
# =====================================================
# PREDICT FUNCTION
# =====================================================

current_model_type = "rf"

def predict(model_type="rf"):
    global current_model_type
    current_model_type = model_type

    if not validate_inputs():
        return

    progress["value"] = 0

    predict_rf_btn.config(state="disabled")
    predict_linear_btn.config(state="disabled")
    reset_btn.config(state="disabled")
    detail_btn.config(state="disabled")

    model_name_label = "Random Forest Regressor" if model_type == "rf" else "Linear Regressor"
    performance_title.config(text=f"📊 Performance ({model_name_label})")

    stages = [
        (f"Loading {model_name_label} Model...", 20),
        ("Reading Historical Dataset...", 40),
        ("Processing Input Features...", 60),
        ("Generating Prediction...", 85),
        ("Finalizing Results...", 100)
    ]

    try:

        for text, value in stages:

            progress_status.config(text=text)

            while progress["value"] < value:

                progress["value"] += 1

                root.update_idletasks()

                root.after(12)

        result = predict_crop(

            state_var.get(),

            district_var.get(),

            crop_var.get(),

            season_var.get(),

            model_type=model_type

        )

        prediction = result["predicted_yield"]


        if prediction >= 3:

            color = SECONDARY

            message = "🌱 Excellent Expected Yield"

        elif prediction >= 2:

            color = WARNING

            message = "🌾 Moderate Expected Yield"

        else:

            color = DANGER

            message = "⚠ Low Expected Yield"


        yield_label.config(

            text=f"{prediction:.3f}",

            fg=color

        )


        prediction_message.config(

            text=(
                f"{message}\n\n"
                f"Model : {result.get('model_type', model_name_label)}\n"
                f"Crop : {crop_var.get()} | District : {district_var.get()}\n"
                f"Season : {season_var.get()}"
            )

        )


        mae_label.config(

            text=f"{result['mae']:.3f}"

        )

        rmse_label.config(

            text=f"{result['rmse']:.3f}"

        )

        r2_label.config(

            text=f"{result['r2']:.3f}"

        )


        progress_status.config(

            text=f"✅ {model_name_label} Prediction Completed Successfully"

        )

        progress["value"] = 100


    except Exception as e:

        progress["value"] = 0

        progress_status.config(

            text="❌ Prediction Failed"

        )

        messagebox.showerror(

            "Prediction Error",

            str(e)

        )

    finally:

        predict_rf_btn.config(state="normal")

        predict_linear_btn.config(state="normal")

        reset_btn.config(state="normal")

        detail_btn.config(state="normal")
# =====================================================
# RESET
# =====================================================

def reset():
    global current_model_type
    current_model_type = "rf"

    state_var.set("")

    district_var.set("")

    crop_var.set("")

    season_var.set("")


    district_combo["values"] = []

    crop_combo["values"] = []

    season_combo["values"] = []


    yield_label.config(

        text="--",

        fg=SECONDARY

    )


    prediction_message.config(

        text="Waiting for prediction..."

    )

    performance_title.config(

        text="📊 Model Performance"

    )

    mae_label.config(text="--")

    rmse_label.config(text="--")

    r2_label.config(text="--")


    progress["value"] = 0


    progress_status.config(

        text="Ready for Prediction"

    )

# =====================================================
# DETAILED VIEW
# =====================================================

def detailed_view():

    if not validate_inputs():
        return

    import detailed_view

    detailed_view.show_window(
        state_var.get(),
        district_var.get(),
        crop_var.get(),
        season_var.get(),
        model_type=current_model_type
    )
# =====================================================
# BUTTON CARD
# =====================================================

button_card = tk.Frame(
    button_bar,
    bg=BACKGROUND
)

button_card.pack(
    fill="x",
    pady=(2,6)
)


def hover_in(e):
    e.widget.config(relief="raised")


def hover_out(e):
    e.widget.config(relief="flat")


button_style = {
    "fg":"white",
    "font":("Segoe UI",10,"bold"),
    "width":17,
    "height":1,
    "cursor":"hand2",
    "relief":"flat",
    "bd":0
}


predict_rf_btn = tk.Button(
    button_card,
    text="🌲 Predict (RF)",
    bg=SECONDARY,
    command=lambda: predict("rf"),
    **button_style
)

predict_rf_btn.pack(
    side="left",
    expand=True,
    padx=5
)


predict_linear_btn = tk.Button(
    button_card,
    text="📈 Predict (Linear)",
    bg="#388E3C",
    command=lambda: predict("linear"),
    **button_style
)

predict_linear_btn.pack(
    side="left",
    expand=True,
    padx=5
)


reset_btn = tk.Button(
    button_card,
    text="🔄 Reset",
    bg=WARNING,
    command=reset,
    **button_style
)

reset_btn.pack(
    side="left",
    expand=True,
    padx=5
)


detail_btn = tk.Button(
    button_card,
    text="📊 Detailed View",
    bg=ACCENT,
    command=detailed_view,
    **button_style
)

detail_btn.pack(
    side="left",
    expand=True,
    padx=5
)


exit_btn = tk.Button(
    button_card,
    text="❌ Exit",
    bg=DANGER,
    command=root.destroy,
    **button_style
)

exit_btn.pack(
    side="left",
    expand=True,
    padx=5
)


for btn in [predict_rf_btn, predict_linear_btn, reset_btn, detail_btn, exit_btn]:

    btn.bind("<Enter>", hover_in)

    btn.bind("<Leave>", hover_out)
# =====================================================
# KEYBOARD SHORTCUTS
# =====================================================

root.bind(
    "<Return>",
    lambda e: predict("rf")
)

root.bind(
    "<Shift-Return>",
    lambda e: predict("linear")
)

root.bind(
    "<Escape>",
    lambda e: reset()
)

root.bind(
    "<F1>",
    lambda e: detailed_view()
)


# =====================================================
# FOOTER
# =====================================================

footer = tk.Frame(
    root,
    bg=PRIMARY,
    height=28
)

footer.pack(
    fill="x",
    side="bottom"
)

footer.pack_propagate(False)

tk.Label(
    footer,
    text="© 2026 AI Crop Yield Prediction System | Developed using Python • Tkinter • Machine Learning",
    bg=PRIMARY,
    fg="white",
    font=("Segoe UI",9)
).pack(
    pady=4
)


# =====================================================
# START
# =====================================================

root.mainloop()
