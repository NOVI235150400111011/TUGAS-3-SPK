import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── Konfigurasi halaman ──────────────────────────────────────
st.set_page_config(
    page_title="DSS Employee Attrition",
    page_icon="👥",
    layout="wide"
)

# ── Load & Train Model (otomatis jika belum ada .pkl) ────────
@st.cache_resource(show_spinner="Memuat model, harap tunggu...")
def load_or_train_model():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.feature_selection import VarianceThreshold

    # Load dataset langsung dari URL publik
    url = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/HR-Employee-Attrition.csv"

    try:
        df_raw = pd.read_csv(url)
    except Exception:
        # Fallback: dataset dari IBM yang tersedia publik
        url2 = "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv"
        df_raw = pd.read_csv(url2)

    # ── Preprocessing ────────────────────────────────────────
    const_cols = [col for col in df_raw.columns if df_raw[col].nunique() == 1]
    id_cols    = ['EmployeeNumber']
    drop_cols  = const_cols + [c for c in id_cols if c in df_raw.columns]

    df = df_raw.drop(columns=drop_cols).copy()
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

    cat_cols    = df.select_dtypes(include='object').columns.tolist()
    binary_cols = [c for c in cat_cols if df[c].nunique() == 2]
    multi_cols  = [c for c in cat_cols if df[c].nunique() > 2]

    le = LabelEncoder()
    for col in binary_cols:
        df[col] = le.fit_transform(df[col])

    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

    X = df.drop(columns=['Attrition'])
    y = df['Attrition']

    sel = VarianceThreshold(threshold=0)
    sel.fit(X)
    X = X[X.columns[sel.get_support()]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)

    return model, list(X_train.columns), df_raw

model, feature_cols, df_raw = load_or_train_model()

# ── Header ───────────────────────────────────────────────────
st.title("👥 Sistem Pendukung Keputusan")
st.subheader("Prediksi Employee Attrition — Random Forest")
st.caption("Kelompok 3 | Sistem Pendukung Keputusan (SI-D)")
st.markdown("---")

# ── Sidebar: Input Data Karyawan ─────────────────────────────
st.sidebar.header("📋 Input Data Karyawan")

def user_input():
    age                   = st.sidebar.slider("Age", 18, 60, 30)
    monthly_income        = st.sidebar.number_input("Monthly Income", 1000, 20000, 5000, step=500)
    distance_from_home    = st.sidebar.slider("Distance From Home (km)", 1, 30, 10)
    years_at_company      = st.sidebar.slider("Years At Company", 0, 40, 5)
    years_in_role         = st.sidebar.slider("Years In Current Role", 0, 18, 3)
    years_since_promotion = st.sidebar.slider("Years Since Last Promotion", 0, 15, 1)
    years_with_manager    = st.sidebar.slider("Years With Current Manager", 0, 17, 3)
    total_working_years   = st.sidebar.slider("Total Working Years", 0, 40, 8)
    num_companies         = st.sidebar.slider("Num Companies Worked", 0, 9, 2)
    training_times        = st.sidebar.slider("Training Times Last Year", 0, 6, 2)

    job_satisfaction         = st.sidebar.selectbox("Job Satisfaction (1=Low, 4=High)", [1,2,3,4], index=2)
    environment_satisfaction = st.sidebar.selectbox("Environment Satisfaction (1=Low, 4=High)", [1,2,3,4], index=2)
    work_life_balance        = st.sidebar.selectbox("Work Life Balance (1=Bad, 4=Best)", [1,2,3,4], index=2)
    job_involvement          = st.sidebar.selectbox("Job Involvement (1=Low, 4=High)", [1,2,3,4], index=2)
    job_level                = st.sidebar.selectbox("Job Level", [1,2,3,4,5], index=1)
    stock_option             = st.sidebar.selectbox("Stock Option Level", [0,1,2,3], index=1)
    performance_rating       = st.sidebar.selectbox("Performance Rating (3=Excellent, 4=Outstanding)", [3,4], index=0)
    education                = st.sidebar.selectbox("Education (1=Below College, 5=Doctor)", [1,2,3,4,5], index=2)
    percent_hike             = st.sidebar.slider("Percent Salary Hike", 11, 25, 14)

    overtime   = st.sidebar.selectbox("OverTime", ["Yes", "No"])
    gender     = st.sidebar.selectbox("Gender", ["Male", "Female"])
    marital    = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    travel     = st.sidebar.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
    department = st.sidebar.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
    edu_field  = st.sidebar.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing",
                                                           "Technical Degree", "Other", "Human Resources"])
    job_role   = st.sidebar.selectbox("Job Role", [
        "Sales Executive", "Research Scientist", "Laboratory Technician",
        "Manufacturing Director", "Healthcare Representative", "Manager",
        "Sales Representative", "Research Director", "Human Resources"
    ])

    daily_rate   = st.sidebar.slider("Daily Rate", 100, 1500, 800)
    hourly_rate  = st.sidebar.slider("Hourly Rate", 30, 100, 65)
    monthly_rate = st.sidebar.slider("Monthly Rate", 2000, 27000, 14000, step=500)

    return pd.DataFrame([{
        "Age": age, "DailyRate": daily_rate, "DistanceFromHome": distance_from_home,
        "Education": education, "EnvironmentSatisfaction": environment_satisfaction,
        "HourlyRate": hourly_rate, "JobInvolvement": job_involvement, "JobLevel": job_level,
        "JobSatisfaction": job_satisfaction, "MonthlyIncome": monthly_income,
        "MonthlyRate": monthly_rate, "NumCompaniesWorked": num_companies,
        "PercentSalaryHike": percent_hike, "PerformanceRating": performance_rating,
        "RelationshipSatisfaction": 3, "StockOptionLevel": stock_option,
        "TotalWorkingYears": total_working_years, "TrainingTimesLastYear": training_times,
        "WorkLifeBalance": work_life_balance, "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": years_in_role, "YearsSinceLastPromotion": years_since_promotion,
        "YearsWithCurrManager": years_with_manager,
        "BusinessTravel": travel, "Department": department, "EducationField": edu_field,
        "Gender": gender, "JobRole": job_role, "MaritalStatus": marital, "OverTime": overtime,
    }])

input_df = user_input()

# ── Preprocessing input ──────────────────────────────────────
def preprocess_input(raw_df):
    from sklearn.preprocessing import LabelEncoder
    df = raw_df.copy()

    binary_map = {
        "OverTime": {"Yes": 1, "No": 0},
        "Gender":   {"Male": 1, "Female": 0}
    }
    for col, mapping in binary_map.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    multi_cols = ["BusinessTravel", "Department", "EducationField", "JobRole", "MaritalStatus"]
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    return df[feature_cols]

processed  = preprocess_input(input_df)
prob       = model.predict_proba(processed)[0]
pred_label = model.predict(processed)[0]
prob_yes   = prob[1]
prob_no    = prob[0]

# ── Metric Cards ─────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🎯 Prediksi", "ATTRITION ⚠️" if pred_label == 1 else "TETAP ✅")
with col2:
    st.metric("📊 Probabilitas Attrition", f"{prob_yes*100:.1f}%")
with col3:
    st.metric("📊 Probabilitas Tetap", f"{prob_no*100:.1f}%")

st.markdown("---")

# ── Gauge & Rekomendasi ──────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("📈 Tingkat Risiko Attrition")
    color = "green" if prob_yes < 0.4 else ("orange" if prob_yes < 0.7 else "red")
    level = "🟢 RENDAH" if prob_yes < 0.4 else ("🟡 SEDANG" if prob_yes < 0.7 else "🔴 TINGGI")

    fig, ax = plt.subplots(figsize=(5, 0.7))
    ax.barh(["Risiko"], [prob_yes], color=color, height=0.4)
    ax.barh(["Risiko"], [1 - prob_yes], left=[prob_yes], color="lightgray", height=0.4)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.4, 0.7, 1.0])
    ax.set_xticklabels(["0%", "40%", "70%", "100%"])
    ax.set_title(f"Probabilitas Attrition: {prob_yes*100:.1f}%  →  {level}")
    ax.spines[["top", "right", "left"]].set_visible(False)
    st.pyplot(fig)

with col_b:
    st.subheader("🔍 Rekomendasi Keputusan")
    if prob_yes < 0.4:
        st.success("✅ Karyawan cenderung **TIDAK akan resign**. Pertahankan kondisi kerja saat ini.")
    elif prob_yes < 0.7:
        st.warning("⚠️ Karyawan berisiko **SEDANG**. Lakukan review kepuasan kerja dan pertimbangkan kenaikan gaji atau promosi.")
    else:
        st.error("🚨 Karyawan berisiko **TINGGI** untuk resign. Segera lakukan intervensi: diskusi karier, penyesuaian kompensasi, atau pengurangan lembur.")

st.markdown("---")

# ── Feature Importance ────────────────────────────────────────
st.subheader("📊 Top 10 Faktor Paling Berpengaruh terhadap Attrition")
importances = pd.Series(
    model.feature_importances_, index=feature_cols
).sort_values(ascending=False).head(10)

fig2, ax2 = plt.subplots(figsize=(9, 4))
importances.plot(kind="barh", ax=ax2, color="steelblue", edgecolor="white")
ax2.invert_yaxis()
ax2.set_xlabel("Importance Score")
ax2.grid(axis="x", alpha=0.4)
st.pyplot(fig2)

# ── Data Input Summary ────────────────────────────────────────
with st.expander("📋 Lihat Data Input Karyawan"):
    st.dataframe(input_df.T.rename(columns={0: "Nilai"}))
