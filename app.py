import os
import warnings

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Estimasi biaya asuransi",
    page_icon=":material/health_and_safety:",
    layout="wide",
)

MODEL_PATH = "insurance_model.pkl"
FEATURES_PATH = "model_features.pkl"
NAME_PATH = "model_nama.pkl"

# Nilai dari notebook
REF_AVG_SMOKER = 32050
REF_AVG_NON_SMOKER = 8440
MAX_CHARGES_IN_DATA = 63770
MODEL_R2 = 0.8148
MODEL_MAE = 2387.63

SEX_OPTIONS = {"Pria": "male", "Wanita": "female"}
SMOKER_OPTIONS = {"Perokok": "yes", "Bukan perokok": "no"}
REGION_LABELS = {
    "northeast": "Timur laut",
    "northwest": "Barat laut",
    "southeast": "Tenggara",
    "southwest": "Barat daya",
}
BMI_LABELS = {
    "underweight": "Berat kurang",
    "normal": "Normal",
    "overweight": "Berat berlebih",
    "obese": "Obesitas",
}
BMI_COLORS = {"underweight": "blue", "normal": "green", "overweight": "orange", "obese": "red"}
FEATURE_LABELS = {
    "age": "Usia",
    "bmi": "BMI",
    "children": "Jumlah anak",
    "sex_male": "Jenis kelamin pria",
    "smoker_yes": "Perokok",
    "region_northwest": "Wilayah barat laut",
    "region_southeast": "Wilayah tenggara",
    "region_southwest": "Wilayah barat daya",
    "bmi_category_obese": "Kategori obesitas",
    "bmi_category_overweight": "Kategori berat berlebih",
    "bmi_category_underweight": "Kategori berat kurang",
}


@st.cache_resource
def load_artifacts():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH)):
        return None, None, None
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURES_PATH)
    model_name = joblib.load(NAME_PATH) if os.path.exists(NAME_PATH) else "Random Forest"
    return model, feature_cols, model_name


def bmi_category(bmi):
    if bmi < 18.5:
        return "underweight"
    elif bmi < 25:
        return "normal"
    elif bmi < 30:
        return "overweight"
    else:
        return "obese"


def build_input_row(age, sex, bmi, children, smoker, region, feature_cols):
    category = bmi_category(bmi)
    row = {
        "age": age,
        "bmi": bmi,
        "children": children,
        "sex_male": 1 if sex == "male" else 0,
        "smoker_yes": 1 if smoker == "yes" else 0,
        "region_northwest": 1 if region == "northwest" else 0,
        "region_southeast": 1 if region == "southeast" else 0,
        "region_southwest": 1 if region == "southwest" else 0,
        "bmi_category_obese": 1 if category == "obese" else 0,
        "bmi_category_overweight": 1 if category == "overweight" else 0,
        "bmi_category_underweight": 1 if category == "underweight" else 0,
    }
    return pd.DataFrame([row]).reindex(columns=feature_cols, fill_value=0)


def predict_charges(model, X):
    # Model dilatih pada log1p(charges), jadi hasilnya dikembalikan dengan expm1
    return float(np.expm1(model.predict(X)[0]))


model, feature_cols, model_name = load_artifacts()

# Sidebar: input peserta
with st.sidebar:
    st.header("Data peserta")
    age = st.slider("Usia", min_value=18, max_value=64, value=30)
    bmi = st.number_input(
        "Indeks massa tubuh (BMI)",
        min_value=10.0,
        max_value=60.0,
        value=25.0,
        step=0.1,
        format="%.1f",
        help="Berat badan (kg) dibagi tinggi badan (m) kuadrat",
    )
    children = st.slider("Jumlah anak", min_value=0, max_value=5, value=0)
    region = st.selectbox(
        "Wilayah",
        options=list(REGION_LABELS),
        format_func=lambda key: REGION_LABELS[key],
    )
    sex_label = st.segmented_control(
        "Jenis kelamin", options=list(SEX_OPTIONS), default="Wanita", required=True
    )
    smoker_label = st.segmented_control(
        "Status merokok", options=list(SMOKER_OPTIONS), default="Bukan perokok", required=True
    )
    st.caption("Ubah nilai di sidebar, hasil akan diperbarui otomatis.")

st.title("Estimasi biaya asuransi kesehatan")
st.caption(f"Prediksi biaya tahunan berbasis machine learning · Model: {model_name or 'tidak diketahui'}")

if model is None:
    st.error(
        "File model belum ditemukan. Letakkan `insurance_model.pkl` dan `model_features.pkl` "
        "di folder yang sama dengan `app.py`. Kedua file dibuat oleh `train.py` atau notebook.",
        icon=":material/error:",
    )
    st.stop()

sex = SEX_OPTIONS[sex_label]
smoker = SMOKER_OPTIONS[smoker_label]
category = bmi_category(bmi)

X = build_input_row(age, sex, bmi, children, smoker, region, feature_cols)
estimate = predict_charges(model, X)
gap_non_smoker = estimate - REF_AVG_NON_SMOKER

# Kartu hasil utama
with st.container(border=True):
    left, right = st.columns([2, 3], gap="large", vertical_alignment="center")

    with left:
        st.caption("Estimasi biaya tahunan")
        st.markdown(f"## ${estimate:,.0f}")
        st.progress(
            min(estimate / MAX_CHARGES_IN_DATA, 1.0),
            text=f"{estimate / MAX_CHARGES_IN_DATA:.0%} dari biaya tertinggi dalam data",
        )
        with st.container(horizontal=True):
            st.badge(
                "Perokok" if smoker == "yes" else "Bukan perokok",
                icon=":material/smoking_rooms:" if smoker == "yes" else ":material/check_circle:",
                color="red" if smoker == "yes" else "green",
            )
            st.badge(f"BMI {BMI_LABELS[category]}", color=BMI_COLORS[category])
            st.badge(f"Wilayah {REGION_LABELS[region]}", icon=":material/location_on:", color="gray")

    with right:
        m1, m2 = st.columns(2)
        m1.metric(
            "Dibanding rata-rata non-perokok",
            f"${REF_AVG_NON_SMOKER:,.0f}",
            delta=round(gap_non_smoker),
            delta_color="inverse",
            border=True,
            height="stretch",
        )
        m2.metric(
            "Kategori BMI",
            BMI_LABELS[category],
            help=f"BMI Anda {bmi:.1f}",
            border=True,
            height="stretch",
        )

tab_compare, tab_factors, tab_model = st.tabs(
    ["Perbandingan", "Faktor pengaruh", "Tentang model"]
)

with tab_compare:
    compare_df = pd.DataFrame(
        {
            "Kelompok": ["Estimasi Anda", "Rata-rata non-perokok", "Rata-rata perokok"],
            "Biaya": [estimate, REF_AVG_NON_SMOKER, REF_AVG_SMOKER],
        }
    )
    base = alt.Chart(compare_df).encode(
        x=alt.X("Kelompok:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Biaya:Q", title="Biaya (USD)"),
        color=alt.Color(
            "Kelompok:N",
            legend=None,
            scale=alt.Scale(
                domain=compare_df["Kelompok"].tolist(),
                range=["#0F766E", "#94A3B8", "#E11D48"],
            ),
        ),
        tooltip=[alt.Tooltip("Kelompok:N"), alt.Tooltip("Biaya:Q", format="$,.0f")],
    )
    bars = base.mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
    labels = base.mark_text(dy=-10, fontWeight="bold").encode(
        text=alt.Text("Biaya:Q", format="$,.0f")
    )
    st.altair_chart((bars + labels).properties(height=320))
    st.caption("Rata-rata diambil dari data latih pada notebook.")

with tab_factors:
    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame(
            {
                "Faktor": [FEATURE_LABELS.get(c, c) for c in feature_cols],
                "Kepentingan": model.feature_importances_,
            }
        ).sort_values("Kepentingan", ascending=False)
        factor_chart = (
            alt.Chart(importance_df)
            .mark_bar(cornerRadiusEnd=4, color="#0F766E")
            .encode(
                x=alt.X("Kepentingan:Q", title="Tingkat kepentingan"),
                y=alt.Y("Faktor:N", sort="-x", title=None),
                tooltip=[alt.Tooltip("Faktor:N"), alt.Tooltip("Kepentingan:Q", format=".3f")],
            )
            .properties(height=360)
        )
        st.altair_chart(factor_chart)
        st.caption(
            "Menunjukkan kontribusi tiap faktor pada model secara umum, bukan pengaruh untuk satu peserta."
        )
    else:
        st.info("Model ini tidak menyediakan informasi tingkat kepentingan fitur.")

with tab_model:
    k1, k2, k3 = st.columns(3)
    k1.metric("Algoritma", model_name or "Tidak diketahui", border=True, height="stretch")
    k2.metric("Skor R²", f"{MODEL_R2:.2f}", border=True, height="stretch")
    k3.metric("Rata-rata error (MAE)", f"${MODEL_MAE:,.0f}", border=True, height="stretch")
    st.markdown(
        "Model dilatih untuk memprediksi logaritma biaya agar distribusinya lebih simetris, "
        "lalu hasilnya dikembalikan ke skala USD. Skor R² dan MAE dihitung pada data uji "
        "yang tidak dipakai saat pelatihan."
    )
    st.caption(
        "Ini estimasi statistik, bukan penawaran resmi atau diagnosis medis. "
        "Untuk polis sebenarnya, hubungi perusahaan asuransi Anda."
    )
